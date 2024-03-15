from transformers import TrainingArguments
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, Optional, Sequence, List



@dataclass
class TrainingArguments(TrainingArguments):

    fixed_traj_eval_hist_len: int = field(default=5, metadata={"help": "The history length of the for the model when"
                                                                       "evaluated on the fixed trajectories dataset"})
    lora_r: int = field(default=32, metadata={"help": "If LoRA is used, its rank"})
    lora_alpha: int = field(default=64, metadata={"help": "If LoRA is used, its alpha value"})
    lora_dropout: float = field(default=0.0, metadata={"help": "If LoRA is used, its dropout value"})
    mm_projector_lr: float = field(default=0.0, metadata={"help": "Only for llava"})

    load_in_8bit: bool = field(default=False, metadata={"Use 8-bit quantization"})
    load_in_4bit: bool = field(default=False, metadata={"Use 8-bit quantization"})
    lora_lm: bool = field(default=False, metadata={"Use Lora for the language model"})

    torch_dtype: str = field(default="bf16", metadata={"The torch dtype, either of 'bf16', 'fp16', 'fp32'"})
    flash_attn_implm: str = field(default="flash_attention_2", metadata={"The attention implementation, one of 'flash_attention_2", 'sdpa'})

    data_path: str = field(default=None, metadata={"Path to the directory containing the training data"})

@dataclass
class GeneralArguments:

    image_model_name_or_path: str = field(
        default="openai/clip-vit-large-patch14",
        metadata="Which image encoder to use. For 'google/vit...' and 'openai/clip...' models LLama-v2 is used as LM."
                 "If a complete VLM is specified, this VLM will be used. Options are openai/clip-vit-large-patch14,"
                 "google/vit-large-patch16-224, Qwen/Qwen-VL, Qwen/Qwen-VL-Chat-Int4, liuhaotian/llava-v1.5-7b,"
                 "liuhaotian/llava-v1.5-13b, Salesforce/instructblip-vicuna-7b")



    llama_acces_token: Optional[str] = field(default=None, metadata="HF access token for LLama")
    llama_checkpoint: Optional[str] = field(default="meta-llama/Llama-2-7b-hf",
                                            metadata="Which LLama checkpoint to use. Only relevant if "
                                                     "model_name_or_path is a vit or clip version")



    fixed_traj_eval_hist_len: int = field(default=5, metadata={"help": "The history length of the for the model when"
                                                                       "evaluated on the fixed trajectories dataset"})


