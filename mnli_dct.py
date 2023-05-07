import argparse
import os
from typing import Tuple
import torch

from data_utils.task_processors import PROCESSORS, load_examples, TRAIN_SET, DEV_SET, HARD_SET, METRICS, DEFAULT_METRICS
from dct.utils import eq_div

from dct.config import TrainConfig, EvalConfig, WrapperConfig
from dct.modeling_dct_mnli import train_dct
import time
import log
logger = log.get_logger('root')
import warnings
warnings.filterwarnings("ignore")
def load_dct_configs(args) -> Tuple[WrapperConfig, TrainConfig, EvalConfig]:

    model_cfg = WrapperConfig(model_name_or_path=args.model_name_or_path,
                              task_name=args.task_name,
                              label_list=args.label_list,
                              max_seq_length=args.dct_max_seq_length,
                              cache_dir=args.cache_dir,
                              output_dir=args.output_dir,
                              embed_size=args.embed_size,
                              do_train=args.do_train,
                              do_eval=args.do_eval)

    train_cfg = TrainConfig(device=args.device,
                            per_gpu_train_batch_size=args.dct_per_gpu_train_batch_size,
                            n_gpu=args.n_gpu,
                            num_train_epochs=args.dct_num_train_epochs,
                            max_steps=args.dct_max_steps,
                            gradient_accumulation_steps=args.dct_gradient_accumulation_steps,
                            weight_decay=args.weight_decay,
                            learning_rate=args.learning_rate,
                            adam_epsilon=args.adam_epsilon,
                            warmup_steps=args.warmup_steps,
                            max_grad_norm=args.max_grad_norm,
                            alpha=args.alpha)

    eval_cfg = EvalConfig(device=args.device,
                          n_gpu=args.n_gpu,
                          metrics=args.metrics,
                          per_gpu_eval_batch_size=args.dct_per_gpu_eval_batch_size)

    return model_cfg, train_cfg, eval_cfg


def main():
    parser = argparse.ArgumentParser(description="Command line interface for P-Tuning.")

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--model_name_or_path", default="albert-xxlarge-v2", type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True, choices=PROCESSORS.keys(),
                        help="The name of the task to train/evaluate on")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--alpha", default=0.9999, type=float,
                        help="Weighting term for the auxiliary language modeling task (only for dct)")
    parser.add_argument("--dct_max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization for dct. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--dct_per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for dct training.")
    parser.add_argument("--dct_per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for dct evaluation.")
    parser.add_argument('--dct_gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass in dct.")
    parser.add_argument("--dct_num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform in dct.")
    parser.add_argument("--dct_max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform in dct. Override num_train_epochs.")

    # Other optional parameters
    parser.add_argument("--train_examples", default=-1, type=int,
                        help="The total number of train examples to use, where -1 equals all examples.")
    parser.add_argument("--eval_examples", default=-1, type=int,
                        help="The total number of test examples to use, where -1 equals all examples.")
    parser.add_argument("--dev_examples", default=-1, type=int,
                        help="The total number of dev examples to use, where -1 equals all examples.")
    parser.add_argument("--split_examples_evenly", action='store_true',
                        help="If true, train examples are not chosen randomly, but split evenly across all labels.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where to store the pre-trained models downloaded from S3.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.1, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=2000, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=289,
                        help="random seed for initialization")
    parser.add_argument('--do_train', action='store_true',
                        help="Whether to perform training")
    parser.add_argument('--do_eval', action='store_true',
                        help="Whether to perform evaluation")
    parser.add_argument("--embed_size", default=768, type=int, help="")

    args = parser.parse_args()
    current_time=str(int(time.time()))
    args.output_dir=args.output_dir+current_time
    logger.info("Parameters: {}".format(args))
    logger.info("current_time: "+current_time)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
            and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count()

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in PROCESSORS:
        raise ValueError("Task '{}' not found".format(args.task_name))
    processor = PROCESSORS[args.task_name]()
    args.label_list = processor.get_labels()


    train_ex_per_label, eval_ex_per_label, dev_ex_per_label = None, None, None
    train_ex, eval_ex, dev_ex = args.train_examples, args.eval_examples, args.dev_examples
    if args.split_examples_evenly:
        train_ex_per_label = eq_div(args.train_examples, len(args.label_list)) if args.train_examples != -1 else -1
        eval_ex_per_label = eq_div(args.eval_examples, len(args.label_list)) if args.eval_examples != -1 else -1
        dev_ex_per_label = eq_div(args.dev_examples, len(args.label_list)) if args.dev_examples != -1 else -1
        train_ex, eval_ex, dev_ex = None, None, None

    train_data = load_examples(
        args.task_name, args.data_dir, TRAIN_SET, num_examples=train_ex, num_examples_per_label=train_ex_per_label)
    eval_data = load_examples(
        args.task_name, args.data_dir, HARD_SET, num_examples=-1, num_examples_per_label=eval_ex_per_label)
    dev_data = load_examples(
        args.task_name, args.data_dir, DEV_SET, num_examples=-1, num_examples_per_label=dev_ex_per_label)

    args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)
    #load args
    dct_model_cfg, dct_train_cfg, dct_eval_cfg = load_dct_configs(args)

    train_dct(eval_data=eval_data,
                  dev_data=dev_data,
                  train_data=train_data,
                  train_config=dct_train_cfg,
                  eval_config=dct_eval_cfg,
                  model_config=dct_model_cfg,
                  output_dir=args.output_dir,
                  do_train=args.do_train,
                  do_eval=args.do_eval,
                  seed=args.seed)

if __name__ == "__main__":
    main()
