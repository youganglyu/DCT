import os
from typing import List, Dict
import log
from dct.config import EvalConfig, TrainConfig
from dct.utils import InputExample,exact_match, save_logits, save_predictions, softmax, LogitsList, set_seed, eq_div
from dct.wrapper_dct_mnli import TransformerModelWrapper
from dct.config import  WrapperConfig
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
logger = log.get_logger('root')




def init_model(config: WrapperConfig) -> TransformerModelWrapper:
    """Initialize a new model from the given config."""
    model = TransformerModelWrapper(config)
    return model


def train_dct(train_data: List[InputExample],
              eval_data: List[InputExample],
              dev_data: List[InputExample],
              model_config: WrapperConfig,
              train_config: TrainConfig,
              eval_config: EvalConfig,
              output_dir: str,
              do_train: bool = False,
              do_eval: bool = False,
              seed: int = 289
              ):
    logger.info("RUN seed {}".format(seed))
    set_seed(seed)
    if do_train:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            wrapper = init_model(model_config)
            train_single_model(train_data, eval_data, dev_data, output_dir, wrapper, train_config, eval_config)
    if do_eval:
        wrapper = init_model(model_config)
        wrapper.only_eval(eval_data, dev_data, eval_config, train_config.n_gpu)

def train_single_model(train_data: List[InputExample],
                       eval_data: List[InputExample],
                       dev_data: List[InputExample],
                       output_dir: str,
                       model: TransformerModelWrapper,
                       config: TrainConfig,
                       eval_config: EvalConfig):
    """
    Train a single model.
    :param model: the model to train
    :param train_data: the training examples to use
    :param config: the training config
    :param eval_config: the evaluation config
    :return: a dictionary containing the global step, average loss and (optionally) results on the train set
    """

    results_dict = {}


    if not train_data:
        logger.warning('Training method was called without training examples')
    else:
        global_step, tr_loss = model.train(
            output_dir=output_dir,
            eval_config=eval_config,
            train_data=train_data,
            dev_data=dev_data,
            eval_data=eval_data,
            per_gpu_train_batch_size=config.per_gpu_train_batch_size,
            n_gpu=config.n_gpu,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            adam_epsilon=config.adam_epsilon,
            warmup_steps=config.warmup_steps,
            max_grad_norm=config.max_grad_norm,
            alpha=config.alpha
        )
        results_dict['global_step'] = global_step
        results_dict['average_loss'] = tr_loss


    return results_dict

#TODO test_model()