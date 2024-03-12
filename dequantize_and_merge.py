import os
import torch
from accelerate import Accelerator
from os.path import join as pjoin

from palme_model_openx import Palme

if __name__ == "__main__":


    # checkpoint_dir_path = "/home/dorka/projects/23/llama-openx/palme/palme/taco_alldata/checkpoint-880/"
    checkpoint_dir_path = "/home/dorka/chkpts/run_openx_test/checkpoint-521/"
    checkpoint_dir_path = "/home/dorka/projects/23/llama-openx/palme/palme/taco_extradata_vitL/checkpoint-352"
    checkpoint_dir_path = "/home/dorka/projects/23/llama-openx/palme/palme/taco_extradata_clipL_single_proj/checkpoint-703"
    # checkpoint_dir_path = "/home/dorka/projects/23/llama-openx/palme/palme/taco_alldata_vitL224_norm_noquant/checkpoint-2642"

    acces_token = "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"
    llama_checkpoint = "meta-llama/Llama-2-7b-hf"

    #checkpoint_image_model = "google/vit-base-patch16-224-in21k"
    checkpoint_image_model = "google/vit-large-patch16-224"
    checkpoint_image_model = "openai/clip-vit-large-patch14"


    # checkpoint_image_model = "openai/clip-vit-base-patch32"

    device_index = Accelerator().process_index
    device_map = {"": device_index}

    model = Palme(llama_checkpoint=llama_checkpoint, acces_token=acces_token, image_model_name=checkpoint_image_model,
                  # config=None,
                  load_in_8bit=True,
                  lora_lm=True,
                  lora_vision=False, freeze_vision=True,
                  device_map=device_map,
                  torch_dtype = torch.bfloat16,
                  )

    ## model.load("/home/dorka/projects/23/llama-openx/palme/palme/taco_alldata/checkpoint-880/pytorch_model.bin")
    print("Load trained model")
    model.load(pjoin(checkpoint_dir_path, "pytorch_model.bin"))
    #
    print("Dequantize model")
    model.lm = model.lm._unload_and_optionally_merge(dtype=torch.bfloat16) # does not work on titan x
    # model.merge_and_unload() # does not work on titan x
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

