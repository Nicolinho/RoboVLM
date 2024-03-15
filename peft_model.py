import os

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_int8_training, \
    set_peft_model_state_dict, PeftModelForCausalLM

from typing import Any, Optional, Union, List

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists, onload_layer
from peft.tuners.lora.bnb import Linear8bitLt

import torch
import bitsandbytes as bnb

from tqdm import tqdm

from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftType,
    _get_submodules,
    get_auto_gptq_quant_linear,
)

class PeftModelForCausalLM(PeftModelForCausalLM):
    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[List[str]] = None,
        dtype=None,
    ):
        if merge:
            if getattr(self.model, "quantization_method", None) == "gptq":
                raise ValueError("Cannot merge LORA layers when the model is gptq quantized")

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    if merge:
                        # target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                        target = self.dequantize_and_merge(target, dtype)
                        if dtype is not None:
                            target.to(dtype)
                    setattr(parent, target_name, target)
                    # self._replace_module(parent, target_name, target.get_base_layer(), target)
                elif isinstance(target, ModulesToSaveWrapper):
                    # save any additional trainable modules part of `modules_to_save`
                    setattr(parent, target_name, target.modules_to_save[target.active_adapter])
            #
            # import gc
            # for obj in gc.get_objects():
            #     try:
            #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #             print(type(obj), obj.size())
            #     except:
            #         pass

            # del target
            torch.cuda.empty_cache()

        return self.model

    @torch.no_grad()
    def dequantize_and_merge(self, quantized_layer, dtype):
        assert isinstance(quantized_layer, Linear8bitLt) # lora subclass

        weight = quantized_layer.base_layer.weight
        state = quantized_layer.base_layer.state

        # import bitsandbytes as bnb
        # weight = quantized_layer.weight
        # state = quantized_layer.state

        if state.SCB is None:
            state.SCB = weight.SCB

        # Dequantize the result of identity matrix and int8 weight because bitsandbytes does not support int8
        # dequantization directly
        im = torch.eye(weight.data.shape[-1]).contiguous().half().to(weight.device)
        im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
        im, Sim = bnb.functional.transform(im, "col32")
        if state.CxB is None:
            state.CxB, state.SB = bnb.functional.transform(weight.data, to_order=state.formatB)
        out32, Sout32 = bnb.functional.igemmlt(im, state.CxB, Sim, state.SB)
        output = bnb.functional.mm_dequant(out32, Sout32, SCim, state.SCB, bias=None).t()

        adapter_names = quantized_layer.active_adapters
        assert len(adapter_names) == 1, "Only one adapter is supported"
        lora_data = quantized_layer.get_delta_weight(adapter_names[0])

        dequant_layer = torch.nn.Linear(output.shape[0], output.shape[1],
                                        bias=quantized_layer.base_layer.bias is not None,
                                        # device=lora_data.device,
                                        device="cpu",
                                        dtype=dtype)




        dequant_layer.weight = torch.nn.Parameter(
            (output.to(lora_data.dtype).to(lora_data.device) + lora_data).to(dtype).contiguous().cpu(),
            requires_grad=quantized_layer.base_layer.weight.requires_grad)
        if quantized_layer.base_layer.bias is not None:
            dequant_layer.bias = torch.nn.Parameter(quantized_layer.base_layer.bias.to(dtype),
                                                    requires_grad=quantized_layer.base_layer.bias.requires_grad)

        return dequant_layer


if __name__ == "__main__":
    from palme_model_openx import Palme
    from accelerate import Accelerator
    from os.path import join as pjoin

    checkpoint_dir_path = None

    acces_token = None # insert if needed
    llama_checkpoint = "meta-llama/Llama-2-7b-hf"

    #checkpoint_image_model = "google/vit-base-patch16-224-in21k"
    checkpoint_image_model = "google/vit-large-patch16-224"

    # checkpoint_image_model = "openai/clip-vit-base-patch32"

    device_index = Accelerator().process_index
    device_map = {"": device_index}

    model = Palme(llama_checkpoint=llama_checkpoint, acces_token=acces_token, image_model_name=checkpoint_image_model,
                  # config=None,
                  load_in_8bit=False,
                  lora_lm=True,
                  lora_vision=False, freeze_vision=True,
                  device_map=device_map,
                  torch_dtype = torch.bfloat16,
                  )

    print("Load trained model")
    model.load(pjoin(checkpoint_dir_path, "pytorch_model.bin"))
    #
    print("Dequantize model")
    # model.lm = model.lm._unload_and_optionally_merge(dtype=torch.bfloat16) # does not work on titan x
    model = model.merge_and_unload() # does not work on titan x
    #
    print("Save dequantized model")
    os.makedirs(pjoin(checkpoint_dir_path, 'dequant'), exist_ok=True)
    torch.save(model.lm.state_dict(),
               pjoin(checkpoint_dir_path, 'dequant', "lm_model.bin"))
    if 'openai/clip' in checkpoint_image_model:
        torch.save(model.proj_layer.state_dict(),
               pjoin(checkpoint_dir_path, 'dequant', "img_proj_layer_model.bin"))
    elif 'google/vit' in checkpoint_image_model:
        torch.save(model.img_embed_model.classifier.state_dict(),
               pjoin(checkpoint_dir_path, 'dequant', "img_embed_model_classifier_model.bin"))

