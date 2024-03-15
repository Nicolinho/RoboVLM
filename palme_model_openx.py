import os
import numpy as np
import torch
from torch import nn

from copy  import deepcopy
from torchvision.transforms import Resize, Compose, Normalize, ToTensor, InterpolationMode
from PIL import Image
from time import time
from collections import deque
from functools import partial

from transformers import get_scheduler, AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM, \
    AutoImageProcessor, DefaultDataCollator, AutoModelForImageClassification, LlamaTokenizer, \
    CLIPModel, CLIPProcessor, AutoModel, CLIPImageProcessor, GPTQConfig, StoppingCriteria, \
    StoppingCriteriaList, BitsAndBytesConfig

from trainer import OpenXTrainer, LLaVATrainer

from peft import LoraConfig, get_peft_model,  prepare_model_for_kbit_training

from training_args import TrainingArguments

# For training
# from peft import PeftModelForCausalLM

#For dequantizing
from peft_model import PeftModelForCausalLM

from accelerate import Accelerator
import evaluate
from datasets import load_dataset, Dataset

from dataset_tools_openx import (generator_fun_openx, generator_fun_openx_nooverlap,
                                 text_to_action, num_proc_to_shard_string, generator_taco_extra_data)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

WANDB_PROJECT="llama-openx"
os.environ["WANDB_PROJECT"] = "llama-openx"

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops):
        StoppingCriteria.__init__(self)
        self.stops = stops.to("cuda")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops = []):
        stop_gen = False
        if input_ids[0].shape[0] >= self.stops.shape[0] and torch.all(input_ids[0][-self.stops.shape[0]:] == self.stops):
            return True
        return False


def find_all_linear_names(model):
    cls = torch.nn.Linear

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def find_all_linear_names_llava(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


class Palme(nn.Module):
    def __init__(self, llama_checkpoint, acces_token, image_model_name, training_args=None, output_dir=None,
                 load_in_8bit=True, load_in_4bit=False, lora_lm=True,
                 quantize_vision=False, lora_vision=False, freeze_vision=True,
                 lora_r=32,  lora_alpha = 64, lora_dropout = 0.05,
                 device_map=None, torch_compile=False, torch_dtype = torch.bfloat16, flash_attn="sdpa"):
        super().__init__()
        assert not (load_in_4bit and load_in_8bit)
        self.freeze_vision = freeze_vision
        self.output_dir = output_dir
        self.lora_lm = lora_lm
        self.lora_vision = lora_vision
        self.quantize_vision = quantize_vision
        self.image_model_name = image_model_name
        self.llama_checkpoint = llama_checkpoint
        if load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type='nf4'
            )
        else:
            quantization_config = None

        if 'openai/clip' in image_model_name:
            processor = CLIPProcessor.from_pretrained(self.image_model_name)
            self.img_transforms = lambda x: processor(images=x, return_tensors="pt", padding=True)['pixel_values']
            self.img_input_size = (processor.image_processor.crop_size['width'],
                                   processor.image_processor.crop_size['height'])
            self.batch_transform_img = True

        if 'BAAI/EVA-CLIP' in image_model_name:
            image_size = 448
            processor = CLIPImageProcessor(size={"shortest_edge": image_size},
                                           # do_center_crop=True,
                                           crop_size=image_size)
            self.img_transforms = processor
            self.img_input_size = (image_size,image_size)
            self.batch_transform_img = True

        # if 'apple/DFN5B-CLIP-ViT-H-14-378' in self.image_model_name:
        #     # _, processor = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384')
        #     _, processor = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-378')
        #
        #     self.img_transforms = processor
        #     self.img_input_size = (378,378) # model hast 384 in string but still size is 378, dunno why
        #     self.batch_transform_img = False

        elif 'google/vit' in image_model_name:
            image_processor = AutoImageProcessor.from_pretrained(self.image_model_name)
            self.image_processor = image_processor
            self.img_input_size = (
                image_processor.size["shortest_edge"]
                if "shortest_edge" in image_processor.size
                else (image_processor.size["height"], image_processor.size["width"]))

            self.img_transforms = lambda x: image_processor(images=x, return_tensors="pt", padding=True)['pixel_values']
            self.batch_transform_img = True

        elif 'Qwen' in image_model_name:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
            from modelling_qwen import QWenLMHeadModel
            load_in_4bit = load_in_8bit = False # Qlora difficult https://github.com/QwenLM/Qwen-VL#q-lora
            if torch_dtype ==  torch.bfloat16:
                load_in_bf16, load_in_fp16 = True, False
            elif torch_dtype ==  torch.float16:
                load_in_bf16, load_in_fp16 = False, True
            else:
                raise Exception("Use Qwen with fp16 or bf16")
            if 'Int4' in image_model_name:
                quantization_config = GPTQConfig(bits=4, disable_exllama=True)
                if load_in_bf16:
                    print("Int4 version of Qwen shoudl be loaded in fp16, set datatype accordingly")
                    load_in_bf16, load_in_fp16 = False, True
            else:
                quantization_config = None
            self.vlm_model = QWenLMHeadModel.from_pretrained(
                image_model_name, quantization_config=quantization_config,
                device_map=device_map, trust_remote_code=True, bf16=load_in_bf16, fp16=load_in_fp16)
            if 'Int4' in image_model_name:
                self.vlm_model = prepare_model_for_kbit_training(self.vlm_model, use_gradient_checkpointing=training_args.gradient_checkpointing)
            mean = (0.48145466, 0.4578275, 0.40821073)
            std = (0.26862954, 0.26130258, 0.27577711)
            image_size = 448
            self.img_transforms = Compose([
                Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                ToTensor(),
                Normalize(mean=mean, std=std)])

            self.img_input_size = None
            self.batch_transform_img = True

        elif 'llava' in self.image_model_name:
            from transformers import LlavaForConditionalGeneration, LlavaConfig, CLIPVisionConfig, LlamaConfig, AutoProcessor
            self.vlm_model = LlavaForConditionalGeneration.from_pretrained(
                self.image_model_name, torch_dtype=torch_dtype, attn_implementation=flash_attn)
            self.vlm_model.to(torch.bfloat16) #TODO
            self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            self.img_transforms = self.processor.image_processor
            tokenizer = self.processor.tokenizer

        elif 'blip' in image_model_name:
            from transformers import InstructBlipProcessor
            from modelling_instructblip import InstructBlipForConditionalGenerationOpenX
            self.vlm_model = InstructBlipForConditionalGenerationOpenX.from_pretrained(image_model_name,
                                                                                  torch_dtype=torch_dtype)
            self.processor = InstructBlipProcessor.from_pretrained(image_model_name)
            tokenizer = self.processor.tokenizer
            self.img_transforms = self.processor.image_processor


        if not any([n in image_model_name for n in ['Qwen', 'llava', 'blip']]):
            tokenizer = LlamaTokenizer.from_pretrained(llama_checkpoint, token=acces_token)
            self.lm = LlamaForCausalLM.from_pretrained(
                llama_checkpoint,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map=device_map,
                token=acces_token,
            )
            emb_size = self.lm.get_input_embeddings().embedding_dim
            self.emb_size = emb_size

            if load_in_8bit or load_in_4bit:
                self.lm = prepare_model_for_kbit_training(self.lm, use_gradient_checkpointing=training_args.gradient_checkpointing)

        if 'Qwen' in image_model_name:
            self.end_action_token_ids = tokenizer.encode(" [ea]") # space important as ' [' is a single token and o/w not recognized when checked for stops
        elif 'llava' in image_model_name:
            self.end_action_token_ids = tokenizer.encode("[ea]")[1:]
        else:
            self.end_action_token_ids = tokenizer.encode("[ea]")[1:]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=torch.tensor(self.end_action_token_ids, dtype=torch.long))])


        if "google/vit" in self.image_model_name:
            self.img_embed_model = AutoModelForImageClassification.from_pretrained(
                image_model_name,
                load_in_8bit=(load_in_8bit and quantize_vision),
                load_in_4bit=(load_in_4bit and quantize_vision),
                quantization_config=quantization_config if ((load_in_4bit or load_in_8bit) and quantize_vision) else None,
                device_map=device_map, torch_dtype=torch_dtype,)

            if freeze_vision:
                for param in (self.img_embed_model.vit.parameters()):
                    param.requires_grad = False

            self.img_embed_model.classifier = nn.Linear(self.img_embed_model.classifier.in_features, emb_size,
                                                        dtype=torch_dtype,
                                                        device=self.img_embed_model.device)
            self.img_embed_model.classifier.weight.data = nn.init.trunc_normal_(
                self.img_embed_model.classifier.weight.data.to(torch.float32), mean=0.0, std=self.img_embed_model.config.initializer_range
                ).to(self.img_embed_model.classifier.weight.dtype)
            self.img_embed_model.classifier.bias.data.fill_(0)

        elif "openai/clip" in self.image_model_name or 'BAAI/EVA-CLIP' in self.image_model_name or \
            'apple/DFN5B-CLIP-ViT-H-14' in self.image_model_name:
            if "openai/clip" in self.image_model_name:
                clip = CLIPModel.from_pretrained(image_model_name,
                                                 load_in_8bit=(load_in_8bit and quantize_vision),
                                                 load_in_4bit=(load_in_4bit and quantize_vision),
                                                 device_map=device_map,
                                                 torch_dtype=torch_dtype,)
            elif 'BAAI/EVA-CLIP' in self.image_model_name or 'apple/DFN5B-CLIP-ViT-H-14-378' in self.image_model_name:
                clip = AutoModel.from_pretrained(
                        image_model_name,
                        load_in_8bit=(load_in_8bit and quantize_vision),
                        load_in_4bit=(load_in_4bit and quantize_vision),
                        quantization_config=quantization_config if (
                                    (load_in_4bit or load_in_8bit) and quantize_vision) else None,
                        device_map=device_map,
                        torch_dtype=torch.bfloat16)

            clip_vision = clip.vision_model
            proj_layer = nn.Linear(clip.vision_embed_dim, emb_size,
                                    dtype=torch.float32,
                                    device=clip.device)
            proj_layer.weight.data = nn.init.trunc_normal_(
                    proj_layer.weight.data.to(torch.float32), mean=0.0, std=clip_vision.config.initializer_range
                    ).to(proj_layer.weight.dtype)
            proj_layer.bias.data.fill_(0)

            self.clip = clip_vision
            self.proj_layer = proj_layer

            self.img_embed_double_proj = False
            if self.img_embed_double_proj:
                self.clip = clip
            else:
                self.clip = clip.vision_model

            self.proj_layer = proj_layer

            if freeze_vision:
                for param in (clip.vision_model.parameters()):
                    param.requires_grad = False


        if lora_lm:
            if 'Qwen' in image_model_name:
                if 'Int4' in image_model_name: # somehow I get nans after first update when lora includes lm_head
                    lora_target_modules = ["c_attn", "attn.c_proj", "w1", "w2"] #https://github.com/QwenLM/Qwen-VL/blob/be7ac08ceaaf67be98ea701ed53138db9205fdd3/finetune.py#L59C33-L59C70
                else:
                    lora_target_modules = ["c_attn", "attn.c_proj", "w1", "w2"]#, 'lm_head'] #https://github.com/QwenLM/Qwen-VL/blob/be7ac08ceaaf67be98ea701ed53138db9205fdd3/finetune.py#L59C33-L59C70
                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                # self.vlm_model = get_peft_model(self.vlm_model, lora_config)
                self.vlm_model = PeftModelForCausalLM(self.vlm_model, lora_config)
                self.vlm_model.print_trainable_parameters()
            elif 'blip' in image_model_name:
                lora_target_modules = find_all_linear_names(self.vlm_model.language_model)
                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                self.vlm_model = PeftModelForCausalLM(self.vlm_model, lora_config)
                self.vlm_model.print_trainable_parameters()
            elif 'llava' in image_model_name:
                lora_target_modules = find_all_linear_names_llava(self.vlm_model)
                # lora_target_modules = ['k_proj', 'linear_2', 'v_proj', 'linear_1', 'q_proj', 'gate_proj', 'up_proj', 'o_proj', 'down_proj']
                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                # self.vlm_model = get_peft_model(self.vlm_model, lora_config)
                self.vlm_model = PeftModelForCausalLM(self.vlm_model, lora_config)

                tune_vision, tune_lm, tune_proj = False, True, False
                if not tune_vision:
                    self.vlm_model.base_model.model.vision_tower.requires_grad_(False)
                    for name, p in self.vlm_model.base_model.model.vision_tower.named_parameters():
                        if 'lora_A' in name or 'lora_B' in name:
                            p *= 0

                if not tune_lm:
                    self.vlm_model.base_model.model.language_model.requires_grad_(False)
                    for name, p in self.vlm_model.base_model.model.language_model.named_parameters():
                        if 'lora_A' in name or 'lora_B' in name:
                            p *= 0

                if not tune_proj:
                    self.vlm_model.base_model.model.multi_modal_projector.requires_grad_(False)
                    for name, p in self.vlm_model.base_model.model.multi_modal_projector.named_parameters():
                        if 'lora_A' in name or 'lora_B' in name:
                            p *= 0

            else:
                lora_target_modules = find_all_linear_names(self.lm)
                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                # self.lm = get_peft_model(self.lm, lora_config)
                self.lm = PeftModelForCausalLM(self.lm, lora_config)
                self.lm.print_trainable_parameters()

        if lora_vision:
            lora_target_modules_vit = find_all_linear_names(self.img_embed_model)
            lora_config_vit = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules_vit,
                lora_dropout=lora_dropout,
                bias="none",
                modules_to_save=["classifier"],
            )
            self.img_embed_model = get_peft_model(self.img_embed_model, lora_config_vit)
            # self.img_embed_model2 = get_peft_model(self.img_embed_model2, lora_config_vit)
            self.img_embed_model.print_trainable_parameters()

        if torch_compile:
            self.do_torch_compile()

        self.tokenizer = tokenizer

    def do_torch_compile(self):
        for id in ['lm', 'vlm_model', 'img_embed_model', 'clip']:
            if hasattr(self, id):
                print("Torch compile ", id)
                setattr(self, id, torch.compile(getattr(self, id)))

    def transforms(self, inputs, padding_strategy=None, max_length=1024):
        t = {}
        if self.batch_transform_img:
            t["images"] = [self.img_transforms([Image.fromarray(np.array(img, dtype=np.uint8)) for img in traj_imgs]) for
                                        traj_imgs in inputs["images"]]
        else:
            t["images"] = [
                torch.stack([self.img_transforms(Image.fromarray(np.array(img, dtype=np.uint8))) for img in traj_imgs]) for
                traj_imgs in inputs["images"]]

        t["image_pos"] = []
        t["text"] = []
        t["labels"] = []
        t["attention_mask"] = []

        instr_dict_token = self.tokenizer(inputs["instruction"])
        t['instruction'] = instr_dict_token["input_ids"]
        tok_instr, mask_instr = deepcopy(instr_dict_token["input_ids"]), deepcopy(instr_dict_token["attention_mask"])
        obs_tokens, act_tokens = [], []
        for i, aseq in enumerate(inputs['actions']):
            tokenized_action_seq = self.tokenizer(aseq)
            tokenized_action_seq_id = tokenized_action_seq["input_ids"]
            tokenized_action_seq_mask = tokenized_action_seq["attention_mask"]
            token_rep = tok_instr[i]
            mask = mask_instr[i]
            labels = [-100] * len(mask)
            image_pos = []
            for ta in range(len(tokenized_action_seq_id)):
                k = len(token_rep)
                if k >= max_length:
                    print("Skip Images as max seq len is reached")
                    t["images"][i] = t["images"][i][:ta,...]
                    break
                token_rep.extend(obs_tokens)
                image_pos.append(len(token_rep)) # placeholder for image
                token_rep.extend([0] + act_tokens) # placeholder for image
                mask.extend([1] * (len(token_rep) - k)) # placeholder for image
                labels.extend([-100] * (len(token_rep) - k)) # placeholder for image
                token_rep.extend(tokenized_action_seq_id[ta][1:] + model.end_action_token_ids) # do not include bos token here
                mask.extend(tokenized_action_seq_mask[ta][1:] + [1] * len(model.end_action_token_ids)) #TODO handle end of action token (eos token might do sth unwanted during training)
                labels.extend(tokenized_action_seq_id[ta][1:] + model.end_action_token_ids)

            if padding_strategy is not None:
                if len(token_rep) > max_length:
                    print(f"Cut seq from {len(token_rep)} to {max_length}")
                    token_rep = token_rep[:max_length]
                    labels = labels[:max_length]
                    mask = mask[:max_length]
                else:
                    token_rep = token_rep + [self.tokenizer.pad_token_id] * (max_length - len(token_rep))
                    labels = labels + [self.tokenizer.pad_token_id] * (max_length - len(labels))
                    mask = mask + [0] * (max_length - len(mask))

            t["text"].append(token_rep)
            t["attention_mask"].append(mask)
            t["labels"].append(labels)
            t["image_pos"].append(image_pos)


        t["labels"] = torch.tensor(t["labels"], dtype=torch.long)
        t["text"] = torch.tensor(t["text"], dtype=torch.long)
        t["attention_mask"] = torch.tensor(t["attention_mask"], dtype=torch.int8)

        bs = len(t["images"])

        # have to bring the sequences with different num of images into same tensor shapes. Assign a -100 to the
        # image_pos corresponding to images which are zero padded
        max_num_imgs = max([images.shape[0] for images in t["images"]])
        images = torch.zeros((bs, max_num_imgs, *t["images"][0].shape[1:]), dtype=t["images"][0].dtype)
        for i, imgs in enumerate(t["images"]):
            images[i,:imgs.shape[0]] = imgs
        t["images"] = images

        image_pos = torch.zeros((bs, max_num_imgs), dtype=torch.long) - 100 # assign -100
        for i, img_pos in enumerate(t["image_pos"]):
            image_pos[i,:len(img_pos)] = torch.LongTensor(img_pos)
        t["image_pos"] = image_pos

        # also pad the instructions
        max_ins_len = max([len(ins) for ins in t["instruction"]])
        t["instruction"] = torch.tensor([ins + [-100] * (max_ins_len - len(ins)) for ins in t["instruction"]], dtype=torch.long)

        return t

    def transforms_qwen(self, inputs, max_length=1024, for_eval=False):
        t = {}
        t["images"] = [torch.stack([self.img_transforms(Image.fromarray(np.array(img, dtype=np.uint8))) for img in traj_imgs]) for
                       traj_imgs in inputs["images"]]

        img_placeholder_string = ''
        model_input = []
        for i in range(len(inputs['actions'])):
            list_for_tok = [{'text': inputs['instruction'][i]}]
            for k in range(len(inputs['actions'][i])):
                list_for_tok.append({'image': img_placeholder_string})
                list_for_tok.append({'text': inputs['actions'][i][k]})
                list_for_tok.append({'text': ' [ea]'}) # the space is important for action decoding right now

            model_input.append(self.tokenizer(self.tokenizer.from_list_format(list_for_tok), return_tensors='pt'))

        t['text'] =  torch.cat([m['input_ids'] for m in model_input], dim=0)
        t['token_type_ids'] =  torch.cat([m['token_type_ids'] for m in model_input], dim=0)
        t['attention_mask'] =  torch.cat([m['attention_mask'] for m in model_input], dim=0)

        assert t['text'].shape[0] == 1, "only bs 1 is supported currently"
        if t['text'].shape[-1] > max_length:
            im_start, im_end = torch.where(t['text'] == 151857)[1], torch.where(t['text'] == 151858)[1]
            last_img_end = torch.where(im_end < max_length)[0]
            cut_idx = im_start[min(torch.max(last_img_end) + 1, im_start.shape[0] - 1)] # could be slightly larger than max len

            print(f"Cut seq from {t['text'].shape[1]} to {cut_idx}")
            t['text'] = t['text'][:, :cut_idx]
            t['token_type_ids'] = t['token_type_ids'][:, :cut_idx]
            t['attention_mask'] = t['attention_mask'][:, :cut_idx]

            cut_img_num = torch.sum(im_end < max_length)
            t["images"][0] = t["images"][0][:cut_img_num, :]


        t["labels"] = torch.where(
            torch.logical_and(torch.logical_and(t['text'] != 151859, t['text'] != 151858), t['text'] != 151857),
            t['text'], -100) # corresponds to the img_pad_id, img_start_id, img_end_id. THis assumes img_placeholder_string = ''
        pos_skip_line_token_after_img_end = torch.where( t['text'] == 151858)[1] + 1
        t["labels"][0, pos_skip_line_token_after_img_end] = -100
        instr_ids = self.tokenizer(inputs['instruction'][i])['input_ids']
        t["labels"][0,:len(instr_ids)] = -100
        # qwen encoding around pictures (between actions) 'Picture', ' ', '1', ':', ' ', '\n',
        # 'Find the nearest electronics store that\'s open tomorrowPicture 1:
        # <img></img>\nInput text "Find the nearest electronics store that\'s open tomorrow" [ea]Picture 2:
        # <img></img>\npress enter [ea]Picture 3:
        # <img></img>\nswipe from 48 71 to 50 62 [ea]Picture 4:
        start_pos = torch.where(t["labels"][0,:] == self.tokenizer("Picture")['input_ids'][0])[0]
        for st in start_pos:
            i = st
            while i < t["labels"].shape[1] and t["labels"][0, i] != -100:
                t["labels"][0,i] = -100
                i += 1

        bs = len(t["images"])
        #  currently only bs 1 for qwen
        t["image_pos"] = torch.zeros((bs, 1))
        t["instruction"] = torch.tensor(self.tokenizer(inputs['instruction'])['input_ids'], dtype=torch.long)



        for k,v in t.items():
            inputs[k] = v

        del inputs['actions']
        if not for_eval:
            del inputs['image_sizes']

        return inputs
    def transforms_llava(self, inputs, for_eval=False):
        t = {}
        imgs = [[Image.fromarray(np.array(img, dtype=np.uint8)) for img in traj_imgs] for traj_imgs in inputs["images"]]

        # "USER: <image>\nturn on the green light ASSISTANT: <answer1>USER: <prompt2>ASSISTANT: <answer2>USER: <prompt3>ASSISTANT:"
        # txt = "USER: <image>\nturn on the green light ASSISTANT: <answer1>USER: <image>ASSISTANT: <answer2>"

        t['text'], t['attention_mask'], t["images"], t['action_pos'] = [], [], [], []
        for i in range(len(inputs['actions'])):
            action_st, action_end = [], []
            text = "USER: <image>\n" + inputs['instruction'][i]
            for j in range(len(inputs['actions'][i])):
                text += " ASSISTANT: "
                action_st.append(len(self.tokenizer.encode(text)))
                text += inputs['actions'][i][j] + " [ea]"
                action_end.append(len(self.tokenizer.encode(text)))
                if j < len(inputs['actions'][i]) - 1:
                    text += "USER: <image>"

            processed = self.processor(text=text, images=imgs[i])
            t['text'].append(processed['input_ids'][0])
            t['attention_mask'].append(processed['attention_mask'][0])
            t['images'].append(processed['pixel_values'])
            t['action_pos'].append(torch.tensor([action_st, action_end], dtype=torch.long))

        t['labels'] = t['text']

        bs = len(t["images"])
        t["image_pos"] = torch.zeros((bs, 1))
        t["instruction"] = torch.tensor(self.tokenizer(inputs['instruction'])['input_ids'], dtype=torch.long)

        for k,v in t.items():
            inputs[k] = v

        del inputs['actions']
        if not for_eval:
            del inputs['image_sizes']

        return inputs

    def transforms_blip(self, inputs, for_eval=False):
        t = {}
        imgs = [[Image.fromarray(np.array(img, dtype=np.uint8)) for img in traj_imgs] for traj_imgs in inputs["images"]]

        # "USER: <image>\nturn on the green light ASSISTANT: <answer1>USER: <prompt2>ASSISTANT: <answer2>USER: <prompt3>ASSISTANT:"
        # txt = "USER: <image>\nturn on the green light ASSISTANT: <answer1>USER: <image>ASSISTANT: <answer2>"

        t['text'], t['attention_mask'], t["images"], t['action_pos'], t['qformer_input_ids'], t['qformer_attention_mask'] = [], [], [], [], [], []
        for i in range(len(inputs['actions'])):
            action_st, action_end = [], []
            text = inputs['instruction'][i]
            for j in range(len(inputs['actions'][i])):
                action_st.append(len(self.tokenizer.encode(text)))
                text += " " + inputs['actions'][i][j] + " [ea]"
                action_end.append(len(self.tokenizer.encode(text)))

            # use only the instruction as query input to the Qformer and not the actions
            processed = self.processor(text=inputs['instruction'][i], images=imgs[i], return_tensors='pt')
            t['qformer_input_ids'].append(processed['qformer_input_ids'][0])
            t['qformer_attention_mask'].append(processed['qformer_attention_mask'][0])
            t['text'].append(self.tokenizer(text, return_tensors='pt')['input_ids'][0])
            t['attention_mask'].append(processed['attention_mask'][0]) #dummy
            t['images'].append(processed['pixel_values'])
            t['action_pos'].append(torch.tensor([action_st, action_end], dtype=torch.long))

        t['labels'] = t['text']

        bs = len(t["images"])
        t["image_pos"] = torch.zeros((bs, 1))
        t["instruction"] = torch.tensor(self.tokenizer(inputs['instruction'])['input_ids'], dtype=torch.long)

        for k,v in t.items():
            inputs[k] = v

        del inputs['actions']
        if not for_eval:
            del inputs['image_sizes']

        return inputs


    def load(self, state_dict_path):
        self.load_state_dict(torch.load(state_dict_path, map_location="cuda"), strict=False)

    def merge_and_unload(self):
        if hasattr(self, 'vlm_model'):
            self.vlm_model = self.vlm_model.merge_and_unload()
        if hasattr(self, 'lm') and self.lora_lm:
            print("merge lora adapter for the LLM")
            self.lm = self.lm.merge_and_unload()
        if hasattr(self, 'img_embed_model') and self.lora_vision:
            print("merge lora adapter for the vision model")
            self.img_embed_model = self.img_embed_model.merge_and_unload()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        print(gradient_checkpointing_kwargs)
        if any([n in self.image_model_name for n in ['Qwen', 'llava', 'blip']]):
            self.vlm_model.enable_input_require_grads()
            self.vlm_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        else:
            self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            self.img_embed_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)



    def embed_inputs(self, text, image, image_pos):
        if "google/vit" in self.image_model_name:
            vit_out = self.img_embed_model.vit(image.flatten(0,1))[0].to(self.img_embed_model.classifier.weight.dtype)
            img_emb = self.img_embed_model.classifier(vit_out[:, 0, :]).reshape([image.shape[0], image.shape[1], -1])
        elif "openai/clip" in self.image_model_name or 'BAAI/EVA-CLIP' in self.image_model_name \
                or 'apple/DFN5B-CLIP-ViT-H-14' in self.image_model_name :
            img_emb = self.proj_layer(self.clip(image.flatten(0,1))["pooler_output"].to(self.proj_layer.weight.dtype)
                                      ).reshape([image.shape[0], image.shape[1], -1])
        input_text_emb = self.lm.get_input_embeddings()(text).clone()
        batch_size = input_text_emb.shape[0]

        for i in range(batch_size):
            non_pad_image_pos = image_pos[i][torch.where(image_pos[i] != -100)] # -100 is the padding value
            input_text_emb[i, non_pad_image_pos, :] = img_emb[i, :non_pad_image_pos.shape[0], ...].to(input_text_emb.dtype)

        return input_text_emb

        #TODO need to merge both embeddings together for to produce one tensor of size (batch_size, sequence_length, hidden_size)

    def forward(self, text, images, image_pos, attention_mask, labels, *arg, **kwargs):
        if 'Qwen' in self.image_model_name:
            return self.vlm_model.forward(input_ids=text, images=images, token_type_ids=kwargs['token_type_ids'],
                                   labels=labels, attention_mask=attention_mask)
        elif 'blip' in self.image_model_name:
            assert images.shape[0] == 1, "only bs 1 is implemeted"
            return self.vlm_model.forward(input_ids=text, pixel_values=images[0], qformer_input_ids=kwargs['qformer_input_ids'],
                                          # qformer_attention_mask=kwargs['qformer_attention_mask'],
                                          action_pos=kwargs['action_pos'])
        elif 'llava' in self.image_model_name:
            return self.vlm_model.forward(input_ids=text, pixel_values=images[0], #TODO batch dim of images
                                          labels=labels, attention_mask=attention_mask)
        else:
            embeddings = self.embed_inputs(text, images, image_pos)
            return self.lm.forward(inputs_embeds=embeddings, labels=labels, attention_mask=attention_mask)

    @torch.no_grad()
    def generate(self, instruction, images, meta_info, return_decoded_actions=True, extract_action=False,
                 gt_history=False, image_pos=None, text=None, action_pos=None, **kwargs):
        #  only works for batch_dim = 1 as otherwise the newly generated predictions that are
        #  then again used as input might differ in length
        assert instruction.shape[0] == 1, "Only generation for batch size = 1 is supported"
        all_generated_ids = None

        if 'Qwen' in self.image_model_name:
            assert gt_history, "Non gt history eval currently not implemented"
            decoded_action_list = []
            img_end_pos = torch.where(text == 151858)[1]
            for i, im_end in enumerate(img_end_pos):
                generation_input_ids = text[:, :im_end+2]
                pred_action_ids = self.vlm_model.generate(
                    inputs=generation_input_ids, images=images, max_new_tokens=35,  stopping_criteria=self.stopping_criteria,
                     temperature=0.95)
                pred_action_ids = pred_action_ids[:, generation_input_ids.shape[1]:]
                all_generated_ids = pred_action_ids.clone() if all_generated_ids is None else \
                    torch.cat((all_generated_ids, pred_action_ids), dim=1)
                if return_decoded_actions:
                    try:
                        action_text = self.tokenizer.batch_decode(pred_action_ids, skip_special_tokens=True,
                                                                  clean_up_tokenization_spaces=False)[0]
                        decoded_action_list.append(text_to_action(action_text, gripper_range_2=True))
                    except Exception as e:
                        print(e)
                        decoded_action_list.append(None)

        elif 'blip' in self.image_model_name:
            assert gt_history, "Non gt history eval currently not implemented"
            decoded_action_list = []
            for i in range(images.shape[1]):
                generation_input_ids = text[:, :action_pos[0, 0, i]]
                pred_action_ids = self.vlm_model.generate(
                    input_ids=generation_input_ids, pixel_values=images[0, :i+1,...], qformer_input_ids=kwargs['qformer_input_ids'],
                    action_pos=action_pos[:, :, :i+1],  max_new_tokens=35,  stopping_criteria=self.stopping_criteria,
                     temperature=0.95)
                # pred_action_ids = pred_action_ids[:, generation_input_ids.shape[1]:]
                all_generated_ids = pred_action_ids.clone() if all_generated_ids is None else \
                    torch.cat((all_generated_ids, pred_action_ids), dim=1)
                if return_decoded_actions:
                    try:
                        action_text = self.tokenizer.batch_decode(pred_action_ids, skip_special_tokens=True,
                                                                  clean_up_tokenization_spaces=False)[0]
                        decoded_action_list.append(text_to_action(action_text, gripper_range_2=True))
                    except Exception as e:
                        print(e)
                        decoded_action_list.append(None)

        elif 'llava' in self.image_model_name:
            assert gt_history, "Non gt history eval currently not implemented"
            decoded_action_list = []
            for i in range(images.shape[1]):
                generation_input_ids = text[:, :action_pos[0, 0, i]]
                pred_action_ids = self.vlm_model.generate(
                    # inputs=generation_input_ids, images=images, max_new_tokens=35,  stopping_criteria=self.stopping_criteria,
                    input_ids=generation_input_ids, pixel_values=images[0,:i+1,...], max_new_tokens=35,  stopping_criteria=self.stopping_criteria,
                     temperature=0.95)
                pred_action_ids = pred_action_ids[:, generation_input_ids.shape[1]:]
                all_generated_ids = pred_action_ids.clone() if all_generated_ids is None else \
                    torch.cat((all_generated_ids, pred_action_ids), dim=1)
                if return_decoded_actions:
                    try:
                        action_text = self.tokenizer.batch_decode(pred_action_ids, skip_special_tokens=True,
                                                                  clean_up_tokenization_spaces=False)[0]
                        decoded_action_list.append(text_to_action(action_text, gripper_range_2=True))
                    except Exception as e:
                        print(e)
                        decoded_action_list.append(None)
        else:

            if gt_history:
                input_text_emb = self.lm.get_input_embeddings()(text).clone()
                all_input_text_emb = self.lm.get_input_embeddings()(text).clone()
            else:
                non_pad_instr = instruction[0][ torch.where(instruction[0] != -100)]
                input_text_emb = self.lm.get_input_embeddings()(non_pad_instr)[None, :]
            if "google/vit" in self.image_model_name:
                vit_out = self.img_embed_model.vit(images.flatten(0,1))[0].to(self.img_embed_model.classifier.weight.dtype)
                img_emb = self.img_embed_model.classifier(vit_out[:, 0, :]).reshape([images.shape[0], images.shape[1], -1]).to(input_text_emb.device)
            elif "openai/clip" in self.image_model_name or 'BAAI/EVA-CLIP' in self.image_model_name or \
                    'apple/DFN5B-CLIP-ViT-H-14' in self.image_model_name:
                img_emb = self.proj_layer(self.clip(images.flatten(0,1))["pooler_output"].to(self.proj_layer.weight.dtype)
                                          ).reshape([images.shape[0], images.shape[1], -1]).to(input_text_emb.device)

            if gt_history:
                non_pad_image_pos = image_pos[0][torch.where(image_pos[0] != -100)] # -100 is the padding value
                all_input_text_emb[0, non_pad_image_pos, :] = img_emb[0, :non_pad_image_pos.shape[0], ...].to(input_text_emb.dtype)

            decoded_action_list = []

            for i in range(images.shape[1]):
                if gt_history:
                    input_text_emb = all_input_text_emb[:, :image_pos[0, i] + 1,:] # include image
                else:
                    input_text_emb = torch.cat((input_text_emb, img_emb[:, i,...].unsqueeze(1)), dim=1)

                sss = time()

                generated_ids = self.lm.generate(inputs_embeds=input_text_emb, max_new_tokens=30,
                                                     stopping_criteria=self.stopping_criteria,
                                                     temperature=0.95,
                                                     # use_cache=True,
                                                     ) #TODO input ids or emb also ok?
                print( time() - sss)

                generated_embeds = self.lm.get_input_embeddings()(generated_ids)
                all_generated_ids = generated_ids.clone() if all_generated_ids is None else \
                    torch.cat((all_generated_ids, generated_ids), dim=1)

                if not gt_history:
                    input_text_emb = torch.cat((input_text_emb, generated_embeds), dim=1)

                if return_decoded_actions:
                    try:
                        action_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                        decoded_action_list.append(text_to_action(action_text, gripper_range_2=True)) #TODO gripper range
                    except Exception as e:
                        print(e)
                        decoded_action_list.append(None)

                print("action gen time: ", time() - last_time)
                last_time = time()

        if return_decoded_actions:
            return all_generated_ids, decoded_action_list
        return all_generated_ids

    def reset_history(self, instruction, max_len=5):
        self.action_step = 0
        self.h_obs = deque([], maxlen=max_len)
        self.h_actions = deque([], maxlen=max_len-1)
        if 'llava' in self.image_model_name:
            self.instr = instruction
        elif 'blip' in self.image_model_name:
            self.instr = instruction
            self.processed_instr = self.processor(text=instruction, return_tensors='pt')
            for k,v in self.processed_instr.items():
                self.processed_instr[k] = v.to(self.vlm_model.device)
        elif 'Qwen' in self.image_model_name:
            self.instr = torch.tensor(self.tokenizer(instruction)["input_ids"], dtype=torch.long, device=self.vlm_model.device)
        else:
            self.instr_emb = self.lm.get_input_embeddings()(torch.tensor(self.tokenizer(instruction)["input_ids"],
                                                                         dtype=torch.long, device=self.lm.device))
        print(self.tokenizer.batch_decode(self.tokenizer(instruction)["input_ids"]))

    @torch.no_grad()
    def select_action(self, image_obs, history_len=5):
        """
        Performs model inference on a given input image and produces an action.
        The image_obs input will be transformed to an PIL image and resized by PIL. Don't resize before with
        another library, as the model is trained with images resized by PIL and there are slight differences between
        resizing algorithms.
        :param image_obs: torch tensor or np array with int8 values in [0,255]
        :param history_len:
        :return:
        """
        self.action_step += 1
        assert type(image_obs) == torch.Tensor or type(image_obs) == np.ndarray
        if type(image_obs) == torch.Tensor:
            image = image_obs.flatten(end_dim=-3).cpu().numpy()

        image = Image.fromarray(image)


        st = time()
        with (torch.no_grad()):
            if not any([n in self.image_model_name for n in ['Qwen', 'llava', 'blip']]):
                if "google/vit" in self.image_model_name:

                    image = self.img_transforms(image).to(self.lm.device)[None, :]
                    vit_out = self.img_embed_model.vit(image)[0].to(
                        self.img_embed_model.classifier.weight.dtype)
                    img_emb = self.img_embed_model.classifier(vit_out[:, 0, :]).reshape([1, -1])

                elif "openai/clip" in self.image_model_name or 'BAAI/EVA-CLIP' in self.image_model_name \
                        or 'apple/DFN5B-CLIP-ViT-H-14' in self.image_model_name:
                    image = self.img_transforms(image).to(self.lm.device)
                    if self.img_embed_double_proj:
                        img_emb = self.proj_layer(
                            self.clip(image.flatten(end_dim=-4))["image_embeds"].to(self.proj_layer.weight.dtype)
                        ).reshape([1, -1]).to(self.instr_emb.device)
                    else:
                        img_emb = self.proj_layer(
                            self.clip(image)["pooler_output"].to(self.proj_layer.weight.dtype)
                            ).reshape([1, -1]).to(self.instr_emb.device)


                self.h_obs.append(img_emb)

                len_act_embds = sum([act_emb.shape[0] for act_emb in self.h_actions])
                len_img_embds = len(self.h_obs)
                lm_input_emb = torch.zeros((1, self.instr_emb.shape[0] + len_act_embds + len_img_embds, self.emb_size), device=self.lm.device,
                                           dtype=self.instr_emb.dtype)
                lm_input_emb[0,:self.instr_emb.shape[0]] = self.instr_emb
                idx = self.instr_emb.shape[0]
                for i in range(len(self.h_actions)):
                    lm_input_emb[0, idx] = self.h_obs[i]
                    idx += 1
                    lm_input_emb[0, idx:idx + len(self.h_actions[i])] = self.h_actions[i]
                    idx += len(self.h_actions[i])
                lm_input_emb[0, -1] = self.h_obs[-1]
                assert idx == lm_input_emb.shape[1] - 1, print(idx, lm_input_emb.shape)
            # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):

            elif 'llava' in self.image_model_name:
                image = self.img_transforms(image, return_tensors='pt')["pixel_values"][0].to(self.vlm_model.device)
                self.h_obs.append(image)
                text = "USER: <image>\n" + self.instr
                for i in range(len(self.h_actions)):
                    text += " ASSISTANT: "
                    text += self.h_actions[i] #+ " [ea]"
                    text += "USER: <image>"

                text += " ASSISTANT: "
                print(text)
                model_input = self.tokenizer(text, return_tensors='pt')['input_ids'].to(self.vlm_model.device)
                images = torch.stack([ im for im in self.h_obs])

            elif 'blip' in self.image_model_name:
                image = self.img_transforms(image, return_tensors='pt')["pixel_values"][0].to(self.vlm_model.device)
                self.h_obs.append(image)
                action_st, action_end = [], []
                text = self.instr
                for i in range(len(self.h_actions)):
                    action_st.append(len(self.tokenizer.encode(text)))
                    text += " " + self.h_actions[i] #+ " [ea]"
                    action_end.append(len(self.tokenizer.encode(text)))

                # to give the start for the action to generate
                action_st.append(len(self.tokenizer.encode(text)))
                text += " "
                action_end.append(len(self.tokenizer.encode(text)))

                action_pos = torch.tensor([[action_st, action_end]], dtype=torch.long)
                model_input = self.tokenizer(text, return_tensors='pt')['input_ids'].to(self.vlm_model.device)
                images = torch.stack([ im for im in self.h_obs])

            else:
                image = self.img_transforms(image).to(self.vlm_model.device)
                self.h_obs.append(image)
                list_for_tok = []
                list_for_tok.append({'image': ''})
                for i in range(len(self.h_actions)):
                    list_for_tok.append({'text': self.h_actions[i]})
                    list_for_tok.append({'image': ''})

                model_input = torch.cat((
                    self.instr,
                    self.tokenizer(self.tokenizer.from_list_format(list_for_tok),
                                   return_tensors='pt')['input_ids'][0].to(self.vlm_model.device)
                    ))

        st = time()

        for i in range(5):
            if 'Qwen' in self.image_model_name:
                generated_ids_all = self.vlm_model.generate(inputs=model_input[None, :], images=torch.stack([ im for im in self.h_obs])[None, :],
                                                 max_new_tokens=25,# max_length=25,
                                                 stopping_criteria=self.stopping_criteria,
                                                 temperature=0.95,
                                                 # use_cache=True,
                                                 )
                generated_ids = generated_ids_all[:, model_input.shape[0]:]
            elif 'llava' in self.image_model_name:
                generated_ids_all = self.vlm_model.generate(input_ids=model_input, pixel_values=images,
                                                 max_new_tokens=25,# max_length=25,
                                                 stopping_criteria=self.stopping_criteria,
                                                 temperature=0.95,
                                                 # use_cache=True,
                                                 )
                generated_ids = generated_ids_all[:, model_input.shape[1]:]
            elif 'blip' in self.image_model_name:
                generated_ids_all = self.vlm_model.generate(input_ids=model_input, pixel_values=images,
                    qformer_input_ids=self.processed_instr['qformer_input_ids'], action_pos=action_pos,
                                                 max_new_tokens=25,# max_length=25,
                                                 stopping_criteria=self.stopping_criteria,
                                                 temperature=0.95,
                                                 )
                generated_ids = generated_ids_all
            else:
                generated_ids = self.lm.generate(inputs_embeds=lm_input_emb, max_new_tokens=25,# max_length=25,
                                                 stopping_criteria=self.stopping_criteria,
                                                 temperature=0.95,
                                                 # use_cache=True,
                                                 )
            st = time()

            try:
                action_text = \
                self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                if self.action_step >= 3 and action_text==self.h_actions[0] and  all([self.h_actions[i] == self.h_actions[i+1] for i in range(len(self.h_actions)-1)]):
                    action_bins = np.array([int(b) for b in action_text.split(" ")[:6]])
                    action_bins_noise = np.random.randint(np.maximum(0, action_bins-2), np.minimum(99, action_bins+2))
                    new_action_text = " ".join([str(b) for b in action_bins_noise] + action_text.split(" ")[-2:])
                    print("Detected prediction of the same action repeatedly, changed action from: ")
                    print('', action_text, "| to:\n", new_action_text)
                    action_text = new_action_text
                action = text_to_action(action_text, gripper_range_2 = True) #TODO depends on the dataset
                if action is None:
                    print("No valid action decoded")
                    pass
                if any([n in self.image_model_name for n in ['Qwen', 'llava', 'blip']]):
                    self.h_actions.append(action_text)
                else:
                    self.h_actions.append(self.lm.get_input_embeddings()(generated_ids)[
                                              0])
                st = time()
                return action

            except Exception as e:
                print("Error in decoding the generated text to a valid action")
                print(e)

        raise Exception("Tried 5 times to generate a valid action and failed. End program.")

def train(model, training_args):
    # ds = Dataset.from_generator(generator_fun_openx,
    #                             gen_kwargs={"builder_dir": "/home/dorka/data/tensorflow_ds/0.1.0",
    #                                         'limit': 10000000,
    #                                         "traj_len": 10},
    #                             num_proc=10, writer_batch_size=5)


    # ds_train = Dataset.from_generator(generator_fun_openx_nooverlap,
    #                         # gen_kwargs={"builder_dir": "/home/dorka/data/tensorflow_ds/fractal20220817_data/0.1.0",
    #                         gen_kwargs={
    #                                     "builder_dir": "/home/dorka/data/tensorflow_ds/taco_play/0.1.0/",
    #                                     # "builder_dir": None,
    #                                     'taco_extra_data_dir': "/home/dorka/data/tensorflow_ds/taco_play/extra_data/taco_extra_processed_15hz_resize/",
    #                                     'limit': 1000000000,
    #                                     # 'limit': 1,
    #                                     'shards': num_proc_to_shard_string(50),
    #                                     'img_resize_dim': model.img_input_size,
    #                                     "traj_len": 10},
    #                         num_proc=50, writer_batch_size=50)
    # # import sys; sys.exit()

    # ds_train = Dataset.from_generator(generator_taco_extra_data,
    #     gen_kwargs={
    #         'data_path': "/home/dorka/data/tensorflow_ds/taco_play/extra_data/taco_extra_processed_15hz_resize/",
    #         # 'limit': 1000000,
    #         # 'shards': num_proc_to_shard_string(10),
    #         # 'img_resize_dim': model.img_input_size,
    #         "traj_len": 10}, #5
    #     num_proc=10, writer_batch_size=50)
    ds_train = load_dataset('/home/dorka/.cache/huggingface/datasets/generator/default-23f268c1c1e3bdbd/0.0.0/')['train']

    # dataset = ds.train_test_split(test_size=0.005, writer_batch_size=5)

    ds_eval = Dataset.from_generator(generator_taco_extra_data,
        gen_kwargs={
            'data_path': "/home/dorka/data/tensorflow_ds/taco_play/extra_data/taco_extra_processed_15hz_resize/",
            # 'limit': 1000000,
            # 'shards': num_proc_to_shard_string(10),
            # 'img_resize_dim': model.img_input_size,
            "traj_len": 10, #5 #in RT they used 15 I think
            "val_split": True, },
        num_proc=10, writer_batch_size=50)

    ds_eval_complete_fixed_traj = Dataset.from_generator(generator_taco_extra_data,
        gen_kwargs={
            "data_path": "/home/dorka/data/tensorflow_ds/taco_play/extra_data/taco_extra_processed_15hz_resize/",
            "traj_len": 2000, "val_split": True, "return_robot_obs": True, "return_unprocessed_actions": True},
        writer_batch_size = 50)

    def img_to_torch(inputs):
        inputs['images'] = torch.tensor(inputs['images'], dtype=torch.uint8)
        inputs['actions_unprocessed'] = np.array(inputs['actions_unprocessed'])
        inputs['robot_obs'] = np.array(inputs['robot_obs'])
        return inputs

    ds_eval_complete_fixed_traj = ds_eval_complete_fixed_traj.with_transform(img_to_torch).remove_columns(['actions'])

    # TODO check
    tokenizer_lm = model.tokenizer
    tokenizer_lm.pad_token = tokenizer_lm.eos_token


    if 'Qwen' in model.image_model_name:
        transform_func = model.transforms_qwen
        max_len = 4096
    elif 'llava' in model.image_model_name:
        transform_func = model.transforms_llava
        max_len = 4096 # has no influence. the image 'tokens' are added in forward path of vlm. each image is mapped to 576 token embeds
    elif 'blip' in model.image_model_name:
        transform_func = model.transforms_blip
        max_len = 4096 # has no influence. the image 'tokens' are added in forward path of vlm. each image is mapped to 32 token embeds
    else:
        transform_func = model.transforms
        max_len = 280
    transform_func_train = partial(transform_func, max_length=max_len)
    transform_func_eval = partial(transform_func, max_length=max_len, for_eval=True)

    ds_train = ds_train.with_transform(transform_func_train)
    ds_eval = ds_eval.with_transform(transform_func_eval)

    data_collator = DefaultDataCollator()


    def compute_metrics(eval_preds):
        rouge_metric = evaluate.load('rouge')
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=-1)
        decoded_preds = tokenizer_lm.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer_lm.pad_token_id)
        decoded_labels = tokenizer_lm.batch_decode(labels, skip_special_tokens=True)
        for i in range(min(3, len(decoded_labels))):
            print(f"\n{'Prediction:':<15}", decoded_preds[i])
            print(f"{'Label:':<15}", decoded_labels[i])
        result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {k: round(v, 4) for k, v in result.items()}
        return result


    if 'llava' in model.image_model_name:
        trainer_cls = LLaVATrainer
    else:
        trainer_cls = OpenXTrainer

    trainer = trainer_cls(
        model=model,
        args=training_args,
        # train_dataset=dataset_train['train'],
        # eval_dataset=dataset_eval['test'],
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        eval_dataset_fixed_traj=ds_eval_complete_fixed_traj,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

if __name__=="__main__":
    # checkpoint_image_model = "google/vit-base-patch16-224-in21k"
    checkpoint_image_model = "google/vit-large-patch16-224"
    # checkpoint_image_model = "openai/clip-vit-large-patch14"
    # checkpoint_image_model = "openai/clip-vit-base-patch32"
    # checkpoint_image_model = "google/vit-base-patch32-384"
    # checkpoint_image_model = "google/vit-large-patch32-384"
    # checkpoint_image_model = "openai/clip-vit-large-patch14-336"
    # checkpoint_image_model = "BAAI/EVA-CLIP-8B"
    # checkpoint_image_model = 'apple/DFN5B-CLIP-ViT-H-14-378'
    # checkpoint_image_model = 'apple/DFN5B-CLIP-ViT-H-14-378'
    checkpoint_image_model = 'Qwen/Qwen-VL'
    # checkpoint_image_model = 'Qwen/Qwen-VL-Chat-Int4' # for 4 bit, use fp16 in this case
    # checkpoint_image_model = 'liuhaotian/llava-v1.5-7b'
    # checkpoint_image_model = 'liuhaotian/llava-v1.5-13b'
    # checkpoint_image_model = 'liuhaotian/llava-v1.6-vicuna-7b'
    checkpoint_image_model = "Salesforce/instructblip-vicuna-7b"
    # checkpoint_image_model = "Salesforce/instructblip-vicuna-13b"
    # openai/clip-vit-large-patch14 for openx
    # try blip2

    per_device_train_batch_size = 1 #4
    per_device_eval_batch_size = 1 #4
    batch_size = 128 #128
    gradient_accumulation_steps = batch_size // per_device_train_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        print("updated gradient_accumulation_steps: ", gradient_accumulation_steps)


    # run_name = 'taco_extradata_vitL_noquant'
    # run_name = 'taco_extradata_qwen_eval'
    run_name = 'taco_extradata_instrblib7b'
    # run_name = 'test'
    # run_name = 'eval_blip'
    # noinspection PyArgumentList
    training_args = TrainingArguments(
        output_dir=run_name,
        run_name=run_name,
        evaluation_strategy="epoch",
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        eval_accumulation_steps = 2,
        fixed_traj_eval_hist_len = 5,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_safetensors=False,
        save_strategy='epoch',
        learning_rate = 3e-4,
        mm_projector_lr = 2e-5, # llava https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune_task_lora.sh
        weight_decay = 0.1,
        adam_beta2 = 0.95,
        max_grad_norm=0.3,
        lr_scheduler_type ="cosine",
        optim="adamw_torch",
        warmup_steps=20,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        num_train_epochs=10,
        logging_steps=1,
        ddp_find_unused_parameters=True,
        report_to="wandb",

    )


    acces_token = "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"
    llama_checkpoint = "meta-llama/Llama-2-7b-hf"

    device_index = Accelerator().process_index
    device_map = {"": device_index}


    model = Palme(llama_checkpoint=llama_checkpoint, acces_token=acces_token, image_model_name=checkpoint_image_model,
                  training_args=training_args, output_dir = training_args.output_dir,
                  load_in_8bit=False, load_in_4bit=False,
                  lora_lm=True, lora_vision=False, freeze_vision=True, quantize_vision=True,
                  lora_r=training_args.lora_r, lora_alpha=training_args.lora_alpha, lora_dropout=training_args.lora_dropout,
                  device_map=device_map,
                  torch_dtype = torch.bfloat16,
                  # torch_dtype = torch.float16,
                  flash_attn = "flash_attention_2", # "sdpa"
                  )


    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available through cuda_visible_devices
        # o/w the batch size is e.g. multiplied with the num of availabel devices
        model.is_parallelizable = True
        model.model_parallel = True

    train(model, training_args)











