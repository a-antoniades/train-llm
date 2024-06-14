#!/usr/bin/env python

"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import torch
from datasets import load_dataset
from datasets import load_from_disk
from datasets import DatasetDict

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# ANTONIS
from datetime import datetime
import json
from omegaconf import OmegaConf
import wandb
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch.distributed as dist
from transformers import get_cosine_schedule_with_warmup
from src.utils import count_tokens
from src._trainer_callbacks import CustomWandbCallback, CustomEvaluationCallback


CACHE_DIR = "/share/edc/home/antonis/datasets/huggingface"
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.31.0.dev0")
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


from transformers import GPTNeoXForCausalLM, GPTNeoXConfig

def calculate_batch_size(batch_size):
    total_batch_size = 1024
    # calculate batch size for each GPU
    n_gpus = torch.cuda.device_count()
    assert (total_batch_size % (n_gpus * batch_size)) == 0, "Total batch size must be divisible by num_gpus * train_micro_batch_size_per_gpu"
    gradient_accumulation_steps = total_batch_size // (n_gpus * batch_size)
    return batch_size


def load_config(config_path):
    with open(config_path, "r") as f:
        config = OmegaConf.load(f)
    return config

def set_training_args(training_args, config):
    paramer_names = [
        'per_device_train',
        'learning_rate',
        'weight_decay',
        
        
    ]

class CustomTrainer(Trainer):
    """
    HF Trainer Wrapper to enforce a cosine lr with warmup
    and a min_lr = 0.1 * max_lr
    """
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)

        # Create a cosine schedule with warmup
        cosine_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=self.args.warmup_steps, 
            num_training_steps=num_training_steps
        )

        # Create a custom scheduler that applies the cosine scheduler and enforces a minimum learning rate
        def custom_scheduler(step):
            lr = cosine_scheduler.get_last_lr()[0]
            min_lr = 0.1 * self.args.learning_rate  # 0.01 times the maximum learning rate
            return max(lr, min_lr)

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, custom_scheduler)
            
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
                "gpt2: 124M; gpt2-medium: 355M; gpt2-large: 774M; gpt2-xl: 1.5B; RWKV/rwkv-4-169m-pile; sgugger/rwkv-430M-pile; sgugger/rwkv-7b-pile."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    rand_init_weights: bool = field(
        default=False,
        metadata={"help": 
                        "Use this to override loading pretrained model weights when you specify"
                        "model_name_or_path. This is useful when you want to train a model from scratch."
                        "but use pretrained tokenizer"}
    )
    resume: Optional[bool] = field(default=False, metadata={"help": "Resume training from checkpoint."})
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
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
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    # train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    # validation_file: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    # )
    # data_config_file: Optional[str] = field(default=None, metadata={"help": "The data config file (a json file)."})
    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The directory of the dataset to use (via the datasets library)."}
    )
    validation_dataset: Optional[str] = field(
        default=None, metadata={"help": "The name of the validation dataset to use (via the datasets library)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    count_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "Count the number of tokens in the datasets and print it. "
                "It's useful to know this number to initialize the model's tokenizer properly."
            )
        },
    )
    limit_total_tokens: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Truncate the number of total tokens in the datasets to this value, "
            )
        },
    )
    save_tokenized_ds: bool = field(
        default=False,
        metadata={
            "help": (
                "Save the tokenized dataset to disk."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    report_every: Optional[int] = field(
        default=50000,
        metadata={"help": "Report training progress every X updates steps."},
    )
    tconf_path: Optional[str] = field(default=None, metadata={"help": "The training config file (a yaml file)."})
    tokenize_only: Optional[bool] = field(default=False)
    wandb_mode: Optional[str] = field(default="run")

    # torch_compile: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to compile the model using torch.jit.script or not."},
    # )
    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

from generate import create_prompt
from transformers import TrainerCallback, TrainerState, TrainerControl

class GenerationEvaluationCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, device, num_return_sequences=1):
        self.eval_dataset = eval_dataset.map(create_prompt)
        self.tokenizer = tokenizer
        self.device = device
        self.num_return_sequences = num_return_sequences

    def on_evaluate(self, args, state, control: TrainerControl, model=None, **kwargs):
        model.eval()  # ensure the model is in evaluation mode
        total_generated_sequences = []

        for idx in range(len(self.eval_dataset)):
            prompt = self.eval_dataset[idx]["prompt"]
            target = self.eval_dataset[idx]["target"]

            # Prepare the prompt for the model
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Perform generation
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    num_return_sequences=self.num_return_sequences,
                    
                )

            # Process the generated sequences
            generated_sequences = [
                self.tokenizer.decode(gen_seq, skip_special_tokens=True)
                for gen_seq in outputs
            ]

            total_generated_sequences.append(generated_sequences)

        # Store the generated sequences in the state for further analysis
        state.log_history["generated_sequences"] = total_generated_sequences

        return control


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    os.environ["WANDB_LOG_MODEL"] = "true"
    os.environ["WANDB_MODE"] = data_args.wandb_mode
    os.environ["WANDB_WATCH"] = "fall"

    # os.environ["WANDB_DISABLED"] = "true"

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    current_time = datetime.now()
    formatted_time = str(current_time.strftime("%d-%m-%y_%H:%M"))
    MODEL_NAME = os.path.join(
                              str(data_args.dataset_dir.split("/")[-3]),
                              str(data_args.dataset_dir.split("/")[-2]),
                              str(data_args.dataset_dir.split("/")[-1]),
                              f"{model_args.model_name_or_path}_ckpt_{model_args.rand_init_weights == False}",
                            )
    training_args.run_name = MODEL_NAME.replace("/", "_")
    training_args.output_dir = os.path.join(training_args.output_dir, MODEL_NAME)
    last_checkpoint = None
    if not model_args.rand_init_weights or model_args.resume:
        if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
    
    # Set seed before initializing model.
    set_seed(training_args.seed)

    os.environ["WANDB_PROJECT"] = "Incidental Supervision - experiment_2"
    os.environ["WANDB_NAME"] = MODEL_NAME.replace("/", "_")
    # if torch.distributed.get_rank() == 0:
    #     wandb.init(project="Incidental Supervision", name=MODEL_NAME, config=vars(training_args),
    #                 group="NLI")

    # Get the datasets
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
    else:
        # Loading a dataset from your local files.
        print(data_args.dataset_dir)
        raw_datasets = {}
        raw_datasets["train"] = load_from_disk(data_args.dataset_dir)
        if data_args.validation_dataset is not None:
            raw_datasets["validation"] = load_from_disk(data_args.validation_dataset)
        else:
            raw_datasets["validation"] = load_from_disk(os.path.join(data_args.dataset_dir, "dataset_validation.arrow"))
        raw_datasets = DatasetDict(raw_datasets)
        data_config_file = os.path.join(data_args.dataset_dir, "config.json")
        print(str(data_config_file))
        file_path = data_config_file
        # json_data_config = json.load(open(file_path))  # Use a different variable name here
        # artifact = wandb.Artifact('data_config', type='data_config', description='data_config', metadata={'data_args': json_data_config})
        # artifact.add_file(data_config_file)
        # wandb.log_artifact(artifact)
            
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if 'pythia' not in model_args.model_name_or_path:
        if model_args.config_name:
            config = AutoConfig.from_pretrained(model_args.config_name)
        # option to load pretrained tokenizer, but not pretrained model
        elif model_args.model_name_or_path and model_args.rand_init_weights is False:
            config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        elif model_args.rand_init_weights is True:
            config = CONFIG_MAPPING[model_args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")
            if model_args.config_overrides is not None:
                logger.info(f"Overriding config: {model_args.config_overrides}")
                config.update_from_string(model_args.config_overrides)
                logger.info(f"New config: {config}")
    else:
        config = GPTNeoXConfig.from_pretrained(model_args.model_name_or_path,
                                               use_cache=False)
        logger.info(f"GPT NEO X config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if "pythia" in model_args.model_name_or_path:
        if model_args.rand_init_weights is False:
            model = GPTNeoXForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                use_cache=False
            )
            print(f"-- n_params: {sum(p.numel() for p in model.parameters())} ----")
        elif model_args.rand_init_weights is True:
            print(config)
            model = AutoModelForCausalLM.from_config(config)
            n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
            logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    else:
        if model_args.model_name_or_path and model_args.rand_init_weights is False:
            torch_dtype = (
                model_args.torch_dtype
                if model_args.torch_dtype in ["auto", None]
                else getattr(torch, model_args.torch_dtype)
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=model_args.low_cpu_mem_usage,
                ignore_mismatched_sizes=True
            )
        elif model_args.rand_init_weights is True:
            model = AutoModelForCausalLM.from_config(config)
            n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
            logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    text_column_name = "text"
    if 'train' in raw_datasets.keys():
        train_column_names = list(raw_datasets["train"].features)
        raw_datasets["train"] = raw_datasets["train"].remove_columns([col for col in raw_datasets["train"].column_names if col != text_column_name])
    else:
        raise ValueError("Train dataset not found in raw_datasets.")
    if 'validation' in raw_datasets.keys():
        val_ds = raw_datasets["validation"]
        if isinstance(val_ds, datasets.dataset_dict.DatasetDict):
            validation_column_names = list(raw_datasets["validation"][list(val_ds.keys())[0]].features)
            # Ensure 'validation' datasets have only the columns that are present in the train datasets
            validation_column_names = [col for col in validation_column_names if col in train_column_names]
            for key in val_ds:
                if text_column_name not in raw_datasets["validation"][key].column_names:
                    raise ValueError(f"Column {text_column_name} not found in validation dataset {key}.")
                raw_datasets["validation"][key] = raw_datasets["validation"][key].remove_columns([col for col in raw_datasets["validation"][key].column_names if col != text_column_name])
        else:
            validation_column_names = list(raw_datasets["validation"].features)
            # Ensure 'validation' datasets have only the columns that are present in the train datasets
            validation_column_names = [col for col in validation_column_names if col in train_column_names]
            if text_column_name not in raw_datasets["validation"].column_names:
                raise ValueError(f"Column {text_column_name} not found in validation dataset.")
            raw_datasets["validation"] = raw_datasets["validation"].remove_columns([col for col in raw_datasets["validation"].column_names  if col != text_column_name])
    else:
        raise ValueError("Validation dataset not found in raw_datasets.")
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    # n_samples = 200
    # raw_datasets['train'] = raw_datasets['train'].select(range(10000))
    # for key in raw_datasets['validation']:
    #     n_samples = min(n_samples, len(raw_datasets['validation'][key]))
    #     raw_datasets['validation'][key] = raw_datasets['validation'][key].select(range(n_samples))

    if isinstance(raw_datasets, datasets.dataset_dict.DatasetDict):
        if not data_args.streaming:
            tokenized_datasets = {x: raw_datasets[x].map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=['text'],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            ) for x in raw_datasets}
        else:
            tokenized_datasets = {x: raw_datasets[x].map(
                tokenize_function,
                batched=True,
                remove_columns=['text'],
            ) for x in raw_datasets}
    else:
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=['text'],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=['text'],
            )
    
        if data_args.count_tokens:
            token_counts = {}
            for key in tokenized_datasets:
                token_counts[key] = count_tokens(tokenized_datasets[key])
            # save as json
            with open(os.path.join(data_args.dataset_dir, "token_counts.json"), "w") as f:
                json.dump(token_counts, f)
        
            logger.info(f"Counted tokens in dataset: {token_counts}")
            exit()

        if data_args.limit_total_tokens is not None:
            from src.utils import limit_total_tokens
            # limit total tokens of TRAIN dataset ONLY
            tokenized_datasets['train'] = limit_total_tokens(tokenized_datasets['train'], data_args.limit_total_tokens)
            save_path = os.path.join(CACHE_DIR, "C4", f"limit_total_tokens_{data_args.limit_total_tokens}")
            tokenized_datasets.save_to_disk(save_path)
            logger.info(f"Saved tokenized datasets to {save_path}")
            exit()
        
        if data_args.save_tokenized_ds is True:
            save_path = os.path.join(data_args.dataset_dir, "tokenized")
            tokenized_datasets.save_to_disk(save_path)
            with open(os.path.join(save_path, "token_counts.json"), "w") as f:
                json.dump(token_counts, f)
            logger.info(f"Saved tokenized datasets to {save_path}")
            exit()
        
    print(tokenized_datasets)

    # # DEBUG
    # # convert tokenized dataset to datasetdict
    # if isinstance(tokenized_datasets, dict):
    #     tokenized_datasets = DatasetDict(tokenized_datasets)
    # # saved tokenized datasets to disk
    # tokenized_datasets.save_to_disk("task_ds_tokenized")
    # exit()

    if data_args.block_size is None:
        # block_size = tokenizer.model_max_length
        block_size = 2048
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        assert total_length != 0, f"total_length is 0, {len(concatenated_examples[list(examples.keys())[0]])}"
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # Remove any example whose input_ids seem to be empty lists
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
    with training_args.main_process_first(desc="grouping texts together"):
        if isinstance(tokenized_datasets, dict) or isinstance(tokenized_datasets, datasets.dataset_dict.DatasetDict):
            if not data_args.streaming:
                lm_datasets = {x: tokenized_datasets[x].map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                ) for x in tokenized_datasets}
                # Assert that all 'input_ids' are the same length as block_size
                # for dataset in lm_datasets.values():
                #     for example in dataset:
                #         # assert len(example) == block_size, "All 'input_ids' should be the same length as block_size"
                #         if example['input_ids'] != block_size:
                #             # discard examples that are not block_size
                #             print(f"Discarding example: {example}")
                #             dataset.remove(example)
                # Filter out examples that are not block_size
                # for key in lm_datasets.keys():
                #     lm_datasets[key] = lm_datasets[key].filter(lambda example: len(example['input_ids']) == block_size if len(example['input_ids']) == block_size else print(f"Dropped example: {example}"))
            else:
                lm_datasets = {x: tokenized_datasets[x].map(
                    group_texts,
                    batched=True,
                ) for x in tokenized_datasets} 
        else:
            if not data_args.streaming:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
            else:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                )
        if data_args.tokenize_only:
            exit()

    
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            labels = labels[:, 1:].tolist()
            preds = preds[:, :-1].tolist()

            # metric1 = evaluate.load("accuracy")
            metric2 = evaluate.load("f1", keep_in_memory=True)
            metric3 = evaluate.load("bleu", keep_in_memory=True)
            metric4 = evaluate.load("meteor", keep_in_memory=True)
            # metric5 = evaluate.load("rouge", keep_in_memory=True)
            # metric6 = evaluate.load('mauve', keep_in_memory=True)
            # metric5 = evaluate.load("bertscore")

            print(f"Predicted: {decoded_preds[0]} \
                  \nTarget: {decoded_labels[0]} ")

            f1 = []
            for i in range(len(labels)):
                # accuracy.append(metric1.compute(predictions=preds[i], references=labels[i]))
                f1.append(metric2.compute(predictions=preds[i], references=labels[i], average='macro')['f1'])
            f1 = np.mean(f1)
            
            # Specifying the average method for multiclass F1 score
            return {
                "f1": f1,
                "bleu": metric3.compute(predictions=decoded_preds, references=decoded_labels),
                "meteor": metric4.compute(predictions=decoded_preds, references=decoded_labels),
                # "rouge": metric5.compute(predictions=decoded_preds, references=decoded_labels),
                # "mauve": metric6.compute(predictions=decoded_preds, references=decoded_labels),
            }

    # assert total batch size is correct
    assert training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps != 240, "Total batch size must be 240"
    # Initialize our Trainer
    if data_args.tconf_path is not None:
        tconf = OmegaConf.load(data_args.tconf_path)
        print(f"// Updating training args with {tconf} //")
        # training_args = TrainingArguments(**tconf)
        # Update the existing TrainingArguments instance
        for key, value in tconf.items():
            setattr(training_args, key, value)
            # Set the minimum learning rate for the scheduler
        if 'pythia' in model_args.model_name_or_path:
            learning_rates = OmegaConf.load(os.path.join(os.path.dirname(data_args.tconf_path), "learning_rates.yml"))
            tconf.learning_rate = learning_rates[model_args.model_name_or_path]


    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        callbacks=[CustomWandbCallback(alert_step=training_args.report_every)] if training_args.report_to == 'wandb' else None,
        data_collator=default_data_collator,
        # compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None
        compute_metrics=None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )
    # Set the minimum learning rate for the scheduler
    if 'pythia' in model_args.model_name_or_path:
        print("hi")
        # print(f"// Setting min_lr to {tconf.learning_rate * 0.01} //")                                                                    
        # trainer.lr_scheduler.min_lrs = [tconf.learning_rate * 0.01]  # replace 0.00001 with your desired min_lr
    # Training
    if training_args.do_train:
        checkpoint = None
        # training_args.resume_from_checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()