import json
import copy
import jsonpickle
import os
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import trange, tqdm
from transformers import  AdamW, get_linear_schedule_with_warmup, \
    BertForMaskedLM, RobertaForMaskedLM, BertConfig, BertTokenizer, RobertaConfig, \
    RobertaTokenizer, AlbertForMaskedLM, AlbertTokenizer, AlbertConfig,BertModel
from transformers.data.metrics import simple_accuracy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

import log
from dct.config import WrapperConfig, EvalConfig
from dct.utils import InputFeatures, DictDataset, distillation_loss, exact_match,InputExample,InputctExample
import torch.nn.functional as F
import random
logger = log.get_logger('root')

def l2norm(x: torch.Tensor):
    norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
    x = torch.div(x, norm)
    return x

class DebiasContrastive(nn.Module):

    def __init__(self, config):
        super(DebiasContrastive, self).__init__()
        self.config = config
        self.num_labels = 3

        config.load_trained_model=False
        if config.load_trained_model:
            self.encoder_q = BertModel(config)
            self.encoder_k = BertModel(config)
        else:
            self.encoder_q = BertModel.from_pretrained(config.model_name_or_path,
            num_labels=len(config.label_list),
            finetuning_task=config.task_name,
            cache_dir=config.cache_dir if config.cache_dir else None,
            use_cache=False,local_files_only=True)
            self.encoder_k = BertModel.from_pretrained(config.model_name_or_path,
            num_labels=len(config.label_list),
            finetuning_task=config.task_name,
            cache_dir=config.cache_dir if config.cache_dir else None,
            use_cache=False,local_files_only=True)

        self.classifier_liner = ClassificationHead(config, self.num_labels)

        self.contrastive_liner_q = ContrastiveHead(config)
        self.contrastive_liner_k = ContrastiveHead(config)

        self.m = 0.999
        self.T = 0.04
        self.train_multi_head = False
        self.multi_head_num = 32

        if not config.load_trained_model:
            self.init_weights()

        # create the label_queue and feature_queue
        self.K = 32000

        self.register_buffer("label_queue", torch.randint(0, self.num_labels, [self.K]))
        self.register_buffer("feature_queue", torch.randn(self.K, 768))
        self.feature_queue = torch.nn.functional.normalize(self.feature_queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.top_k = 1
        self.end_k = 150
        self.update_num = 16

        self.memory_bank = False
        self.random_positive = False

    def _dequeue_and_enqueue(self, keys, label):
        # TODO 我们训练过程batch_size是一个变动的，每个epoch的最后一个batch数目后比较少，这里需要进一步修改
        # keys = concat_all_gather(keys)
        # label = concat_all_gather(label)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.K:
            head_size = self.K - ptr
            head_keys = keys[: head_size]
            head_label = label[: head_size]
            end_size = ptr + batch_size - self.K
            end_key = keys[head_size:]
            end_label = label[head_size:]
            self.feature_queue[ptr:, :] = head_keys
            self.label_queue[ptr:] = head_label
            self.feature_queue[:end_size, :] = end_key
            self.label_queue[:end_size] = end_label
        else:
            # replace the keys at ptr (dequeue ans enqueue)
            self.feature_queue[ptr: ptr + batch_size, :] = keys
            self.label_queue[ptr: ptr + batch_size] = label

        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def select_pos_neg_sample(self, liner_q: torch.Tensor, label_q: torch.Tensor):
        label_queue = self.label_queue.clone().detach()        # K
        feature_queue = self.feature_queue.clone().detach()    # K * hidden_size

        # 1、将label_queue和feature_queue扩展到batch_size * K
        batch_size = label_q.shape[0]
        tmp_label_queue = label_queue.repeat([batch_size, 1])
        tmp_feature_queue = feature_queue.unsqueeze(0)
        tmp_feature_queue = tmp_feature_queue.repeat([batch_size, 1, 1]) # batch_size * K * hidden_size

        # 2、计算相似度
        cos_sim = torch.einsum('nc,nkc->nk', [liner_q, tmp_feature_queue])

        # 3、根据label取正样本和负样本的mask_index
        tmp_label = label_q.unsqueeze(1)
        tmp_label = tmp_label.repeat([1, self.K])

        pos_mask_index = torch.eq(tmp_label_queue, tmp_label)
        neg_mask_index = ~ pos_mask_index

        # 4、根据mask_index取正样本和负样本的值
        feature_value = cos_sim.masked_select(pos_mask_index)
        pos_sample = torch.full_like(cos_sim, -np.inf).cuda()
        pos_sample = pos_sample.masked_scatter(pos_mask_index, feature_value)

        feature_value = cos_sim.masked_select(neg_mask_index)
        neg_sample = torch.full_like(cos_sim, -np.inf).cuda()
        neg_sample = neg_sample.masked_scatter(neg_mask_index, feature_value)

        # 5、取所有的负样本和前top_k 个正样本， -M个正样本（离中心点最远的样本）
        pos_mask_index = pos_mask_index.int()
        pos_number = pos_mask_index.sum(dim=-1)
        pos_min = pos_number.min()
        if pos_min == 0:
            return None
        pos_sample, _ = pos_sample.topk(pos_min, dim=-1)
        pos_sample_top_k = pos_sample[:, 0:self.top_k]
        pos_sample_last = pos_sample[:, -self.end_k:]
        # pos_sample_last = pos_sample_last.view([-1, 1])

        pos_sample = torch.cat([pos_sample_top_k, pos_sample_last], dim=-1)
        pos_sample = pos_sample.view([-1, 1])

        neg_mask_index = neg_mask_index.int()
        neg_number = neg_mask_index.sum(dim=-1)
        neg_min = neg_number.min()
        if neg_min == 0:
            return None
        neg_sample, _ = neg_sample.topk(neg_min, dim=-1)
        neg_sample = neg_sample.repeat([1, self.top_k + self.end_k])
        neg_sample = neg_sample.view([-1, neg_min])
        logits_con = torch.cat([pos_sample, neg_sample], dim=-1)
        logits_con /= self.T
        return logits_con

    def select_pos_neg_random(self, liner_q: torch.Tensor, label_q: torch.Tensor):
        label_queue = self.label_queue.clone().detach()        # K
        feature_queue = self.feature_queue.clone().detach()    # K * hidden_size

        # 1、将label_queue和feature_queue扩展到batch_size * K
        batch_size = label_q.shape[0]
        tmp_label_queue = label_queue.repeat([batch_size, 1])
        tmp_feature_queue = feature_queue.unsqueeze(0)
        tmp_feature_queue = tmp_feature_queue.repeat([batch_size, 1, 1]) # batch_size * K * hidden_size

        # 2、计算相似度
        cos_sim = torch.einsum('nc,nkc->nk', [liner_q, tmp_feature_queue])

        # 3、根据label取正样本和负样本的mask_index
        tmp_label = label_q.unsqueeze(1)
        tmp_label = tmp_label.repeat([1, self.K])

        pos_mask_index = torch.eq(tmp_label_queue, tmp_label)
        neg_mask_index = ~ pos_mask_index

        # 4、根据mask_index取正样本和负样本的值
        feature_value = cos_sim.masked_select(pos_mask_index)
        pos_sample = torch.full_like(cos_sim, -np.inf).cuda()
        pos_sample = pos_sample.masked_scatter(pos_mask_index, feature_value)

        feature_value = cos_sim.masked_select(neg_mask_index)
        neg_sample = torch.full_like(cos_sim, -np.inf).cuda()
        neg_sample = neg_sample.masked_scatter(neg_mask_index, feature_value)

        # 5、取所有的负样本和随机取N个正样本
        pos_mask_index = pos_mask_index.int()
        pos_number = pos_mask_index.sum(dim=-1)
        pos_min = pos_number.min()
        if pos_min == 0:
            return None

        pos_range = [index for index in range(pos_min)]
        pos_index = random.sample(pos_range, self.top_k)
        pos_sample, _ = pos_sample.topk(pos_min, dim=-1)
        pos_sample = pos_sample[:, pos_index]
        # pos_sample_last = pos_sample[:, -self.end_k:]
        # pos_sample_last = pos_sample_last.view([-1, 1])

        # pos_sample = torch.cat([pos_sample_top_k, pos_sample_last], dim=-1)
        pos_sample = pos_sample.view([-1, 1])

        neg_mask_index = neg_mask_index.int()
        neg_number = neg_mask_index.sum(dim=-1)
        neg_min = neg_number.min()
        if neg_min == 0:
            return None
        neg_sample, _ = neg_sample.topk(neg_min, dim=-1)
        neg_sample = neg_sample.repeat([1, self.top_k])
        neg_sample = neg_sample.view([-1, neg_min])
        logits_con = torch.cat([pos_sample, neg_sample], dim=-1)
        logits_con /= self.T
        return logits_con

    def init_weights(self):
        for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
            param_k.data = param_q.data

    def update_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.contrastive_liner_q.parameters(), self.contrastive_liner_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def reshape_dict(self, batch):
        for k, v in batch.items():
            shape = v.shape
            batch[k] = v.view([-1, shape[-1]])
        return batch

    def forward(self,
                query,
                positive_sample=None,
                negative_sample=None,
                negtive_sample_labels=None,
                ):
        labels = query["labels"]
        labels = labels.view(-1)

        #TODO 添加困难负样本的label
        if not self.memory_bank:
            with torch.no_grad():
                self.update_encoder_k()
                update_sample = self.reshape_dict(positive_sample)
                bert_output_p = self.encoder_k(**update_sample)
                update_keys = bert_output_p[1]
                update_keys = self.contrastive_liner_k(update_keys)
                update_keys = l2norm(update_keys)
                tmp_labels = labels.unsqueeze(-1)
                tmp_labels = tmp_labels.repeat([1, self.update_num])
                tmp_labels = tmp_labels.view(-1)
                self._dequeue_and_enqueue(update_keys, tmp_labels)
                #negative
                bert_output_p = self.encoder_k(**negative_sample)
                update_keys = bert_output_p[1]
                update_keys = self.contrastive_liner_k(update_keys)
                update_keys = l2norm(update_keys)
                self._dequeue_and_enqueue(update_keys, negtive_sample_labels)
        query.pop("labels")
        bert_output_q = self.encoder_q(**query)
        q = bert_output_q[1]
        liner_q = self.contrastive_liner_q(q)
        liner_q = l2norm(liner_q)
        logits_cls = self.classifier_liner(q)

        if self.num_labels == 1:
            loss_fct = MSELoss()
            loss_cls = loss_fct(logits_cls.view(-1), labels)
        else:
            loss_fct = CrossEntropyLoss()
            loss_cls = loss_fct(logits_cls.view(-1, self.num_labels), labels)

        if self.random_positive:
            logits_con = self.select_pos_neg_random(liner_q, labels)
        else:
            logits_con = self.select_pos_neg_sample(liner_q, labels)

        if logits_con is not None:
            labels_con = torch.zeros(logits_con.shape[0], dtype=torch.long).cuda()
            loss_fct = CrossEntropyLoss()
            loss_con = loss_fct(logits_con, labels_con)

            loss = loss_con * 0.1 + \
                   loss_cls * (1-0.1)
        else:
            loss = loss_cls

        return loss


    # 考虑eval过程写在model内部？
    def predict(self, query):
        with torch.no_grad():
            bert_output_q = self.encoder_q(**query)
            q = bert_output_q[1]
            logits_cls = self.classifier_liner(q)
            contrastive_output = self.contrastive_liner_q(q)
        return contrastive_output, logits_cls

    def get_features(self, query):
        with torch.no_grad():
            bert_output_k = self.encoder_k(**query)
            contrastive_output = self.contrastive_liner_k(bert_output_k[1])
        return contrastive_output

    def update_queue_by_bert(self,
                             inputs=None,
                             labels=None
                             ):
        with torch.no_grad():
            update_sample = self.reshape_dict(inputs)
            roberta_output = self.encoder_k(**update_sample)
            update_keys = roberta_output[1]
            tmp_labels = labels.unsqueeze(-1)
            tmp_labels = tmp_labels.view(-1)
            self._dequeue_and_enqueue(update_keys, tmp_labels)


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels):
        super(ClassificationHead, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(768, num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ContrastiveHead(nn.Module):
    def __init__(self, config):
        super(ContrastiveHead, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(768, 768)

    def forward(self, feature):
        x = self.dropout(feature)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class TransformerModelWrapper:
    """A wrapper around a Transformer-based language model."""

    def __init__(self, config: WrapperConfig):
        self.config = config

        tokenizer_class = BertTokenizer
        self.tokenizer = tokenizer_class.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir if config.cache_dir else None)
        self.model = DebiasContrastive(config)
        if self.config.do_eval:
            self.model.encoder_q = BertModel.from_pretrained(config.model_name_or_path)
            save_path_file = os.path.join(config.model_name_or_path, "embeddings.pth")
            data = torch.load(save_path_file)
            self.model.classifier_liner.load_state_dict(data["mlp"])
        self.negative_data = None
        self.features = None

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.cuda()

    def create_negative_dataset(self,data):
        negative_dataset = {}
        negative_dataset[0]=[]
        negative_dataset[1] = []
        negative_dataset[2] = []
        for i in range(len(data["labels"])):
            if data["labels"][i]==0:
                temp_dict={}
                temp_dict['input_ids']=data["input_ids"][i]
                temp_dict['attention_mask'] = data["attention_mask"][i]
                temp_dict['token_type_ids'] = data["token_type_ids"][i]
                negative_dataset[0].append(temp_dict)
            elif data["labels"][i]==1:
                temp_dict = {}
                temp_dict['input_ids'] = data["input_ids"][i]
                temp_dict['attention_mask'] = data["attention_mask"][i]
                temp_dict['token_type_ids'] = data["token_type_ids"][i]
                negative_dataset[1].append(temp_dict)
            elif data["labels"][i]==2:
                temp_dict = {}
                temp_dict['input_ids'] = data["input_ids"][i]
                temp_dict['attention_mask'] = data["attention_mask"][i]
                temp_dict['token_type_ids'] = data["token_type_ids"][i]
                negative_dataset[2].append(temp_dict)
        return negative_dataset

    def save(self, path: str) -> None:
        logger.info("Saving models.")
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.encoder_q.save_pretrained(path)

        self.tokenizer.save_pretrained(path)
        state = {"mlp": model_to_save.classifier_liner.state_dict()}
        save_path_file = os.path.join(path, "embeddings.pth")
        torch.save(state, save_path_file)

    def train(self,
              train_data:List[InputctExample],
              eval_data:List[InputExample],
              dev_data:List[InputExample],
              eval_config:EvalConfig,
              output_dir,
              per_gpu_train_batch_size: int = 8,
              n_gpu: int = 1,
              num_train_epochs: int = 3,
              gradient_accumulation_steps: int = 1,
              weight_decay: float = 0.0,
              learning_rate: float = 3e-5,
              adam_epsilon: float = 1e-8,
              warmup_steps=0,
              max_grad_norm: float = 1,
              logging_steps: int = 50,
              max_steps=-1, **_):
        """
        Train the underlying language model.

        :param train_data: the training examples to use
        :param per_gpu_train_batch_size: the number of training examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param num_train_epochs: the number of epochs to train
        :param gradient_accumulation_steps: the number of gradient accumulation steps before performing an update
        :param weight_decay: the weight decay to use
        :param learning_rate: the learning rate to use
        :param adam_epsilon: epsilon parameter for the Adam optimizer
        :param warmup_steps: the number of warmup steps
        :param max_grad_norm: the maximum norm for the gradient
        :param logging_steps: the number of steps after which logging information is printed
        :param max_steps: the maximum number of training steps, overrides ``num_train_epochs``
        :return: a tuple consisting of the total number of steps and the average training loss
        """

        train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
        train_dataset = self._generate_dataset_train_begin(train_data)

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (max(1, len(train_dataloader) // gradient_accumulation_steps)) + 1
        else:
            t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

        print("\n")
        print("num_steps_per_dataset:")
        num_steps=len(train_dataloader) // gradient_accumulation_steps
        print(len(train_dataloader) // gradient_accumulation_steps)
        print("num_train_epochs:")
        num_train_epochs=5
        print(num_train_epochs)
        print("total_steps:")
        t_total=num_steps*num_train_epochs
        print(t_total)
        print("\n")


        cur_model = self.model.module if hasattr(self.model, 'module') else self.model
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=t_total)

        best_global_step = 0
        best_loss = 0.0


        global_step = 0
        self.model.zero_grad()

        train_iterator = trange(int(num_train_epochs), desc="Epoch")

        num_epoch = 0
        batch_num = len(train_dataloader)
        for _ in train_iterator:
            train_dataset = self._generate_dataset_train_epoch(num_epoch)
            # train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
            tr_loss, logging_loss = 0.0, 0.0
            num_epoch += 1
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):

                self.model.train()
                batch = {k: t.cuda(non_blocking=True) for k, t in batch.items()}
                # print(batch)
                # exit()
                # batch
                loss = self.train_step(batch)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                tr_loss += loss.detach().cpu().numpy()
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    # writer.add_scalar("train_loss", (tr_loss - prev_loss), global_step=global_step)
                    prev_loss = tr_loss

                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    # torch.nn.utils.clip_grad_norm_(self.mlp.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    # if logging_steps > 0 and global_step % logging_steps == 0:
                    #     logs = {}
                    #     loss_scalar = (tr_loss - logging_loss) / logging_steps
                    #     learning_rate_scalar = scheduler.get_lr()[0]
                    #     logs['learning_rate'] = learning_rate_scalar
                    #     logs['loss'] = loss_scalar
                    #     logging_loss = tr_loss
                    #     print(json.dumps({**logs, **{'step': global_step}}))

                    # if global_step % self.config.eval_every_step == 0:
            learning_rate_scalar = scheduler.get_lr()[0]
            logger.info(
                "Epoch {} Loss {:.2f} BERT_lr {}".format(num_epoch, tr_loss / batch_num, learning_rate_scalar))
            # epoch evaluation
            dev_scores = self.eval_dev(dev_data, eval_config, n_gpu)
            self.eval_val(eval_data, eval_config, n_gpu)
            logger.info("model save: " + output_dir + '/epoch_' + str(num_epoch))
            self.save(output_dir + '/epoch_' + str(num_epoch))
            # state = {'optimizer': optimizer.state_dict()}
            # save_path_file = os.path.join(output_dir + '/epoch_' + str(num_epoch), "opt.pth")
            # torch.save(state, save_path_file)

        return best_global_step, (best_loss / best_global_step if best_global_step > 0 else -1)

    def only_eval(self,
              eval_data: List[InputExample],
              dev_data: List[InputExample],
              eval_config: EvalConfig,n_gpu: int = 1):
        dev_scores = self.eval_dev(dev_data, eval_config, n_gpu)
        self.eval_val(eval_data, eval_config, n_gpu)

    def eval_dev(self, dev_data, eval_config, n_gpu):
        self.model.eval()
        results = self.eval(dev_data,
                            per_gpu_eval_batch_size=eval_config.per_gpu_eval_batch_size,
                            n_gpu=n_gpu)
        # prompt_name = ['logits_rw', 'logits_poe', 'logits_dt']
        # for pt_name in prompt_name:
        #     logger.info(pt_name)
        #     predictions = np.argmax(results[pt_name], axis=1)
        predictions = np.argmax(results['logits'], axis=1)
        scores = {}
        metrics = eval_config.metrics if eval_config.metrics else ['acc']
        for metric in metrics:
            if metric == 'acc':
                scores[metric] = simple_accuracy(predictions, results['labels'])
            elif metric == 'f1':
                scores[metric] = f1_score(results['labels'], predictions)
            elif metric == 'f1-macro':
                scores[metric] = f1_score(results['labels'], predictions, average='macro')
            elif metric == 'em':
                scores[metric] = exact_match(predictions, results['labels'], results['question_ids'])
            else:
                raise ValueError(f"Metric '{metric}' not implemented")
        logger.info("MNLI dev: {:.2f}%".format(scores['acc'] * 100))
        logger.info('---')
        # logger.info("MNLI ACCU {:.2f}%".format(scores['acc']*100))
        return scores

    def eval_val(self, dev_data, eval_config, n_gpu):
        self.model.eval()
        results = self.eval(dev_data,
                            per_gpu_eval_batch_size=eval_config.per_gpu_eval_batch_size,
                            n_gpu=n_gpu)
        # prompt_name = ['logits_rw', 'logits_poe', 'logits_dt']
        # for pt_name in prompt_name:
        #     logger.info(pt_name)
        #     predictions = np.argmax(results[pt_name], axis=1)
        predictions = np.argmax(results['logits'], axis=1)
        scores = {}
        metrics = eval_config.metrics if eval_config.metrics else ['acc']
        for metric in metrics:
            if metric == 'acc':
                scores[metric] = simple_accuracy(predictions, results['labels'])
            elif metric == 'f1':
                scores[metric] = f1_score(results['labels'], predictions)
            elif metric == 'f1-macro':
                scores[metric] = f1_score(results['labels'], predictions, average='macro')
            elif metric == 'em':
                scores[metric] = exact_match(predictions, results['labels'], results['question_ids'])
            else:
                raise ValueError(f"Metric '{metric}' not implemented")
        logger.info("snli val: {:.2f}%".format(scores['acc'] * 100))
        logger.info('---')
        # logger.info("MNLI ACCU {:.2f}%".format(scores['acc']*100))
        return scores

    def eval(self,
             eval_data: List[InputExample],
             per_gpu_eval_batch_size: int = 8,
             n_gpu: int = 1) -> Dict:

        eval_dataset = self._generate_dataset_eval(eval_data)
        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        # preds_rw = None
        # preds_poe = None
        # preds_dt = None
        preds = None
        all_indices, out_label_ids, question_ids = None, None, None
        eval_losses = [0.0]

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = {k: t.cuda() for k, t in batch.items()}
            labels = batch['labels']
            with torch.no_grad():
                    logits = self.eval_step(batch)
            if preds is None:
                # preds_rw = logits_rw.detach().cpu().numpy()
                # preds_poe = logits_rw.detach().cpu().numpy()
                # preds_dt = logits_rw.detach().cpu().numpy()
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                # preds_rw = np.append(preds_rw, logits_rw.detach().cpu().numpy(), axis=0)
                # preds_poe = np.append(preds_poe, logits_poe.detach().cpu().numpy(), axis=0)
                # preds_dt = np.append(preds_dt, logits_dt.detach().cpu().numpy(), axis=0)
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
        return {
            "eval_loss": np.mean(eval_losses),
            'logits': preds,
            'labels': out_label_ids,
        }

    def _generate_dataset_train_begin(self, data: List[InputExample], labelled: bool = True):
        features = self._convert_examples_to_features(data, labelled=labelled)
        #add hard neg
        hard_index = np.load('./pos_neg_index/SNLI/hard_index_snli.npy')
        hard_features = []
        for i in range(len(features)):
            if i in hard_index:
                hard_features.append(features[i])

        dct_features = features + hard_features
        self.features = dct_features
        feature_dict = {
            'input_ids': torch.tensor([f.input_ids for f in dct_features], dtype=torch.long),
            'attention_mask': torch.tensor([f.attention_mask for f in dct_features], dtype=torch.long),
            'token_type_ids': torch.tensor([f.token_type_ids for f in dct_features], dtype=torch.long),
            'labels': torch.tensor([f.labels for f in dct_features], dtype=torch.long),
        }
        hard_features_dict = {
            'input_ids': torch.tensor([f.input_ids for f in hard_features], dtype=torch.long),
            'attention_mask': torch.tensor([f.attention_mask for f in hard_features], dtype=torch.long),
            'token_type_ids': torch.tensor([f.token_type_ids for f in hard_features], dtype=torch.long),
            'labels': torch.tensor([f.labels for f in hard_features], dtype=torch.long),
        }
        # self.contrastive_dataset=hard_features_dict
        self.negative_data = self.create_negative_dataset(hard_features_dict)
        #construct extract dataset for pos and neg
        return DictDataset(**feature_dict)

    def _generate_dataset_train_epoch(self, epoch):
        features=self.features
        mnli_hard_neg_index=np.load('./pos_neg_index/SNLI/snli_hard_tbert_negindex_epoch'+str(epoch+1)+'.npy')
        feature_dict = {
            'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            'labels': torch.tensor([f.labels for f in features], dtype=torch.long),
        }
        # self.train_dataset=self.create_hard_negative_dataset(feature_dict)
        #TODO 添加数据集样本和其对应的负样本
        dyneg_data=[]
        for i in range(len(mnli_hard_neg_index)):
            dyneg_data.append(features[mnli_hard_neg_index[i]])
        feature_dict['n_input_ids']=torch.tensor([f.input_ids for f in dyneg_data], dtype=torch.long)
        feature_dict['n_attention_mask']=torch.tensor([f.attention_mask for f in dyneg_data], dtype=torch.long)
        feature_dict['n_token_type_ids']=torch.tensor([f.token_type_ids for f in dyneg_data], dtype=torch.long)
        feature_dict['n_labels'] = torch.tensor([f.labels for f in dyneg_data], dtype=torch.long)


        #construct extract dataset for pos and neg
        return DictDataset(**feature_dict)

    def _generate_dataset_eval(self, data: List[InputExample], labelled: bool = True):
        features = self._convert_examples_to_features(data, labelled=labelled)
        feature_dict = {
            'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            'labels': torch.tensor([f.labels for f in features], dtype=torch.long),
        }
        return DictDataset(**feature_dict)

    def _convert_examples_to_features(self, examples: List[InputExample], labelled: bool = True) -> List[InputFeatures]:
        VERBALIZER = {
            "entailment": 0,
            "contradiction": 1,
            "neutral": 2
        }
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example {}".format(ex_index))
            input_features = self.tokenizer(example.text_a, example.text_b, padding='max_length', truncation=True,
                                            max_length=256)
            input_features['labels'] = VERBALIZER[example.label]
            features.append(input_features)
            """
            if ex_index < 5:
                logger.info(f'--- Example {ex_index} ---')
                logger.info(input_features.pretty_print(self.tokenizer))
            """
        return features

    def generate_default_inputs_train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        input_ids = batch['input_ids']

        inputs_bias = {'attention_mask': batch['attention_mask'], 'input_ids': input_ids}
        inputs_bias['token_type_ids'] = batch['token_type_ids']
        inputs_bias['labels']=batch['labels']

        return inputs_bias

    def generate_default_inputs_eval(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        input_ids = batch['input_ids']

        inputs_bias = {'attention_mask': batch['attention_mask'], 'input_ids': input_ids}
        inputs_bias['token_type_ids'] = batch['token_type_ids']

        return inputs_bias


    def reshape_dict(self, sample_num, batch):
        for k, v in batch.items():
            shape = v.shape
            batch[k] = v.view([-1, sample_num, shape[-1]])
        return batch

    @staticmethod
    def list_item_to_tensor(inputs_list: List[Dict]):
        batch_list = {}
        for key, value in inputs_list[0].items():
            batch_list[key] = []
        for inputs in inputs_list:
            for key, value in inputs.items():
                batch_list[key].append(value)

        batch_tensor = {}
        for key, value in batch_list.items():
            # logger.info(value)
            # logger.info(type(value[0]))
            # batch_tensor[key] = torch.tensor(value)
            # logger.info(len(value))
            batch_size=len(value)
            batch_tensor[key] = torch.cat(value,dim=0)
            # logger.info(batch_tensor[key].shape)
            batch_tensor[key]=batch_tensor[key].reshape(batch_size,-1)
            # logger.info(batch_tensor[key].shape)

        return batch_tensor

    def generate_positive_sample(self, label: torch.Tensor):
        # positive_num = self.args.positive_num

        positive_num = 16
        positive_sample = []
        for index in range(label.shape[0]):
            input_label = int(label[index])
            positive_sample.extend(random.sample(self.negative_data[input_label], positive_num))
        # logger.info(type(positive_sample))
        return self.reshape_dict(positive_num, self.list_item_to_tensor(positive_sample))

    # def generate_negative_sample(self, neg_index: torch.Tensor):
    #     # positive_num = self.args.positive_num
    #     negative_sample=[]
    #     # negative_sample = {'input_ids':[],'attention_mask':[],'token_type_ids':[]}
    #     for index in range(neg_index.shape[0]):
    #         input_neg = int(neg_index[index])
    #         # negative_sample['input_ids'].append(self.train_dataset[input_neg]['input_ids'])
    #         # negative_sample['attention_mask'].append(self.train_dataset[input_neg]['attention_mask'])
    #         # negative_sample['token_type_ids'].append(self.train_dataset[input_neg]['token_type_ids'])
    #         negative_sample.append(self.train_dataset[input_neg])


        # return self.list_item_to_tensor(negative_sample)

    def train_step(self, labeled_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a training step."""

        #query and positive_sample

        inputs = self.generate_default_inputs_train(labeled_batch)
        labels = labeled_batch['labels']
        batch_neg_labels=labeled_batch['n_labels']


        #inputs bert用的batch ** 和 label


        # negative_sample = None

        positive_sample = self.generate_positive_sample(labels)
        positive_sample = {k: t.cuda(non_blocking=True) for k, t in positive_sample.items()}
        negative_sample= {'input_ids': labeled_batch['n_input_ids'],
            'attention_mask': labeled_batch['n_attention_mask'],
            'token_type_ids': labeled_batch['n_token_type_ids'],
        }
        # logger.info("positive")
        # logger.info(positive_sample['input_ids'].shape)
        # logger.info("negative")
        # logger.info(negative_sample['input_ids'].shape)
        # logger.info(batch_neg_labels.shape)

        loss = self.model(inputs, positive_sample, negative_sample,batch_neg_labels)

        return loss

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a evaluation step."""
        inputs_bias = self.generate_default_inputs_eval(batch)

        bert_output_q = self.model.encoder_q(**inputs_bias)
        q = bert_output_q[1]
        logits_cls = self.model.classifier_liner(q)

        return logits_cls

