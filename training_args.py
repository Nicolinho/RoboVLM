from transformers import TrainingArguments
from dataclasses import asdict, dataclass, field, fields



@dataclass
class TrainingArguments(TrainingArguments):

    fixed_traj_eval_hist_len: int = field(default=5, metadata={"help": "The history length of the for the model when"
                                                                       "evaluated on the fixed trajectories dataset"})
    lora_r: int = field(default=32, metadata={"help": "If LoRA is used, its rank"})
    lora_alpha: int = field(default=64, metadata={"help": "If LoRA is used, its alpha value"})
    lora_dropout: float = field(default=0.0, metadata={"help": "If LoRA is used, its dropout value"})
    mm_projector_lr: float = field(default=0.0, metadata={"help": "Only for llava"})

