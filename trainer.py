
import math
import torch
from tqdm import tqdm
import numpy as np
from torch import nn
from transformers.trainer import Trainer
from transformers.utils import logging
from transformers.trainer_utils import EvalPrediction
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.get_logger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


class MyTrainer(Trainer):

    def __init__(self, model, train_dataset, eval_dataset, args, tokenizer):
        super(MyTrainer, self).__init__(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset, args=args)
        self.tokenizer = tokenizer

    def train(self, **kwargs):
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        t_total = int(num_update_steps_per_epoch * self.args.num_train_epochs)
        self.args.max_steps = t_total
        self.create_optimizer_and_scheduler(num_training_steps=t_total)

        model = self.model

        logger.info("***** Running training *****")

        epochs_trained = 0
        num_train_epochs = math.ceil(self.args.num_train_epochs)

        tr_loss = 0
        eval_step = num_update_steps_per_epoch / 5
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            step = 0
            for batch in tqdm(epoch_iterator, desc="Iteration", ncols=80):
                cur_loss = self.three_task_train(model, batch)
                tr_loss += cur_loss

                self.optimizer.step()
                self.lr_scheduler.step()
                model.zero_grad()
                step += 1
                if step % eval_step == 0:
                    result = self.evaluate(model=model)
                    print(result)
                    self.save_model(model, self.args.output_dir + '/' + str(epoch) + '_' + str(step) + '/')
                print('{:d}/{:d}, loss={:.4f}\r'.format(step, len(epoch_iterator), cur_loss), end='')

    def save_model(self, model, output_dir):
        logger.info("Saving model checkpoint to %s", output_dir)
        if hasattr(model, "module"):
            model_to_save = model.module
        else:
            model_to_save = model
        model_to_save.save_pretrained(output_dir)

    def defense_train_step(self, model, batch):
        model.train()
        inputs = self._prepare_inputs(batch)  # to device here

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        labels = inputs['labels']

        loss = model(input_ids, attention_mask, token_type_ids, labels)

        # only return the loss item 
        # loss already backward in model-forward
        loss = loss.mean()

        return loss.item()

    def prediction_step(self,model, inputs, ):
        model.eval()
        inputs = self._prepare_inputs(inputs)  # to device here
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        labels = inputs['labels']

        # run on a single-gpu # need to notice what is going on here
        loss, cls_scores, token_scores, class_labels, output_labels = model.module.forward_inference(input_ids, attention_mask, token_type_ids, labels)

        return loss, cls_scores, token_scores, class_labels, output_labels

    def evaluate(self, model=None, eval_dataset=None, ):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        task_1_preds = []
        task_1_label_ids = []
        with torch.no_grad():
            for batch in tqdm(eval_dataloader):
                inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask'],
                          'token_type_ids': batch['token_type_ids'], 'labels':batch['labels']}
                labels = batch['labels']
                outputs_ = self.prediction_step(model, inputs)
                logits = outputs_[1]
                task_1_preds.append(logits.cpu())
                task_1_label_ids.append(torch.tensor(labels).cpu())

            task_1_preds = torch.cat(task_1_preds).cpu().numpy()
            task_1_label_ids = torch.cat(task_1_label_ids).cpu().numpy()
            metrics_task_1 = self.compute_metrics(EvalPrediction(predictions=task_1_preds, label_ids=task_1_label_ids))
            # print(metrics_task_1)
            metrics = {'acc': metrics_task_1['acc']}
            print(metrics)
            return metrics
