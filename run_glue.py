
import argparse
import torch
import torch.nn as nn
import logging
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from transformers import EvalPrediction
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from transformers import GlueDataTrainingArguments as DataTrainingArguments
from data.datasets.glue import GlueDataset
from data.processors.glue import glue_output_modes, glue_tasks_num_labels
from data.processors.utils import glue_compute_metrics
from transformers import AutoTokenizer, AutoConfig
from trainer import MyTrainer
from model import BertAT, RobustBert

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None
    )
    w2i: Optional[str] = field(
        default=None
    )
    cos_mat: Optional[str] = field(
        default=None
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args))

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    args.n_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    if args.no_cuda:
        device = torch.device('cpu')
        args.n_gpu = 0
    args.device = device

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    # )

    model = BertAT.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    model = model.to(device)

    # target_model = RobustBert.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    # )
    # target_model.to(device)
    target_model = model

    # Get datasets
    train_dataset = GlueDataset(data_args, tokenizer=tokenizer,
                                cache_dir=model_args.cache_dir) if training_args.do_train else None
    eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev",
                               cache_dir=model_args.cache_dir) if training_args.do_eval else None

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            if output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            else:  # regression
                preds = np.squeeze(preds)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    # Initialize our Trainer

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    trainer.compute_metrics = build_compute_metrics_fn(data_args.task_name)

    # Training
    if training_args.do_train:
        trainer.train()
        if hasattr(model, "module"):
            model_to_save = model.module
        else:
            model_to_save = model
        trainer.save_model(model_to_save, training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_result = trainer.evaluate_three_task()

        output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
            for key, value in eval_result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))
            eval_results.update(eval_result)

    return eval_results


if __name__ == "__main__":
    main()
