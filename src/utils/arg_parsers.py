from dataclasses import dataclass, field
from typing import Optional
from transformers.utils import add_start_docstrings
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    llama: bool = field(default=False, metadata={"help": "Llama model"})


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_datas_path_pattern: str = field(
        metadata={"help": "The input training data path pattern."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )


@dataclass
# @add_start_docstrings(TrainingArguments.__doc__)
class TrainingArguments():
    max_length: int = field(
        default=512,
        metadata={"help": "input max_length."},
    )
    gradient_accumulation_steps: int = field(
        default=2,
        metadata={"help": "gradient_accumulation_steps."},
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "per_device_train_batch_size."},
    )
    num_iterations: int = field(
        default=50000,
        metadata={
            "help": "Total number of steps to run the model for."
        },
    )
    local_rank: int = field(
        default=0,
        metadata={
            "help": "local_rank for dist."
        },
    )
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "gradient_checkpointing"}
    )
    log_steps: int = field(
        default=10,
        metadata={
            "help": "Print logs after the config steps. Defaults to 10."
        },
    )
    check_point_steps: int = field(
        default=10,
        metadata={
            "help": "Save checkpoint after the config steps. Defaults to 10."
        },
    )
    checkpoint_dir: str = field(
        default='./checkpoint_dir',
        metadata={
            "help": "The base experiment directory to save experiments to."
        },
    )
    ds_config: Optional[str] = field(
        default="/workspace/projects/llm/config/deepspeed_config_stage3.json",
        metadata={
            "help": "pass the path to deepspeed json config file (e.g. `ds_config.json`)"
        },
    )