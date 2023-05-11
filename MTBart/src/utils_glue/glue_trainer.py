import os.path
from transformers import Trainer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from utils_glue.glue_utils import seed_everything, ProgressBar, compute_metrics
from transformers import AutoModelForSequenceClassification
#import matplotlib.pyplot as plt
import numpy as np
#import pynvml
#import pandas as pd
#import seaborn as sns


class bertTrainer(object):
    global_step = 0
    best_step = None
    best_score = .0
    best_score2 = .0
    cnt_patience = 0

    def __init__(
            self,
            args,
            model,
            logger,
            tokenizer,
            metric,
            train_dataset=None,
            train_dataloader=None,
            eval_datasets=None,
            eval_dataloader=None,
            data_collator=None,
    ):
        self.args = args
        self.model = model
        self.metric=metric
        self.tokenizer=tokenizer
        self.train_dataset = train_dataset
        self.eval_datasets = eval_datasets
        self.eval_dataloader = None
        self.train_dataloader = train_dataloader
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_optimizer_and_scheduler(self, num_training_steps: int):
        args = self.args
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=num_training_steps,
                                                    num_warmup_steps=num_training_steps * args.warmup_ratio)
        return optimizer, scheduler

    def train(self):
        args = self.args
        logger = self.logger
        num_training_steps = len(self.train_dataloader) * args.num_train_epochs
        optimizer, scheduler = self._get_optimizer_and_scheduler(num_training_steps)
        seed_everything(args.seed)
        self.model.zero_grad()

        logger.info("***** Running training *****")
        logger.info("Num samples %d", len(self.train_dataloader.dataset))
        logger.info("Num epochs %d", args.num_train_epochs)
        logger.info("Num training steps %d", num_training_steps)
        logger.info("Num warmup steps %d", num_training_steps * args.warmup_ratio)

        self.global_step = 0
        self.best_step = None
        self.best_score = .0
        self.best_score2 = .0
        self.cnt_patience = 0

        for i in range(int(args.num_train_epochs)):
            self.train_epoch(self.train_dataloader, optimizer, scheduler)

        logger.info("Training Stop! The best step %s: %s", self.best_step, self.best_score)
        if self.best_score2>0:
            logger.info("The mnli-mm best step %s: %s", self.best_step, self.best_score2)
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        self._save_best_checkpoint(best_step=self.best_step)
        return self.global_step, self.best_step

    def train_epoch(self, train_dataloader, optimizer, scheduler):
        args = self.args

        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')

        for step, item in enumerate(train_dataloader):
            model = self.model.to(self.device)
            loss = self.training_step(model, item)
            pbar(step, {'loss': loss.item()})
            '''
            if (step + 1) % getattr(args, "accumulate_steps", 1) == 0:
                if args.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            '''
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            self.global_step += 1

            if args.logging_steps > 0 and self.global_step % args.logging_steps == 0:
                print("")
                scores = self.evaluate(model)
                score = scores[0] if type(scores)==list else scores
                score2 = scores[1] if type(scores)==list else 0
                if score >= self.best_score:
                    self.best_score = score
                    self.best_score2 = score2
                    self.best_step = self.global_step
                    self.cnt_patience = 0
                    self._save_checkpoint(model, self.global_step)
                    #self.model = self.model.from_pretrained(os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}"))

        return 0

    def training_step(self, model, item):
        raise NotImplementedError

    def evaluate(self, model):
        raise NotImplementedError

    def _save_checkpoint(self, model, step):
        raise NotImplementedError

    def _save_best_checkpoint(self, best_step):
        raise NotImplementedError


class GLUETrainer(bertTrainer):
    def __init__(
        self,
        args,
        model,
        logger,
        metric,
        tokenizer,
        train_dataset=None,
        train_dataloader=None,
        eval_datasets=None,
        eval_dataloader=None,
        data_collator=None,
        task_name = None,
    ):
        super(GLUETrainer, self).__init__(
            args=args,
            model=model,
            train_dataset=train_dataset,
            train_dataloader=train_dataloader,
            eval_datasets=eval_datasets,
            eval_dataloader=eval_dataloader,
            logger=logger,
            metric=metric,
            tokenizer=tokenizer,
        )
        self.data_collator = data_collator
        self.task_name = task_name
    def training_step(self, model, item):
        model.train()
        inputs = {k: v.to(self.device) for k, v in item.items()}
        outputs = model(**inputs) ##?? inputs里面有哪些东西
        loss = outputs[0]
        loss.backward()
        return loss.detach()

    def evaluate(self, model):
        args = self.args
        logger = self.logger
        tasks = [self.task_name]
        ans = []
        if self.task_name == "mnli":
            tasks.append("mnli-mm")
        for eval_dataset, task in zip(self.eval_datasets, tasks):
            eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.per_device_eval_batch_size,
                               collate_fn=self.data_collator)
            #eval_dataloader = Trainer.get_eval_dataloader(eval_dataset)
            num_examples = len(eval_dataset)

            preds = None
            eval_labels = None
            logger.info("***** Running evaluation *****")
            logger.info("Num samples %d", num_examples)
            for step, item in enumerate(eval_dataloader):
                model.eval()
                inputs = {k: v.to(self.device) for k, v in item.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                    loss, logits = outputs[:2]

                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    eval_labels = inputs['labels'].detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    eval_labels = np.append(
                        eval_labels, inputs['labels'].detach().cpu().numpy(), axis=0
                    )

            task_name = args.output_dir.split('/')[-2]
            preds = np.squeeze(preds) if task_name == 'STSB' else np.argmax(preds, axis=1)
            metrics = compute_metrics(preds, eval_labels, self.metric)
            logger.info("%s-%s acc: %s", task, metrics)

            if task_name == 'CoLA':
                #ans.append(metrics['matthews_correlation'])
                return metrics['matthews_correlation']
            if task_name == 'STSB':
                #ans.append(metrics['combined_score'])
                return metrics['combined_score']
            if task_name == 'MRPC' or task_name == 'QQP':
                #ans.append(metrics['f1'])
                return metrics['f1']
            if task_name == 'MNLI':
                ans.append(metrics['accuracy'])

        if self.task_name == "mnli":
            return ans
        else:
            return metrics['accuracy']





    def predict(self, test_dataset, model):
        args = self.args
        logger = self.logger
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=self.args.per_device_eval_batch_size,
                               collate_fn=self.data_collator)
        num_examples = len(test_dataloader)
        model.to(self.device)
        preds = None
        logger.info("***** Running prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(test_dataloader), desc="Prediction")
        for step, item in enumerate(test_dataloader):
            model.eval()
            inputs = {k: v.to(self.device) for k, v in item.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                #logits = outputs[1].detach()
                loss, logits = outputs[:2]
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            pbar(step=step, info="")

        task_name = args.output_dir.split('/')[-2]
        preds = np.squeeze(preds) if task_name == 'STSB' else np.argmax(preds, axis=1)
        return preds


    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, "checkpoint-{}".format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        self.tokenizer.save_vocabulary(save_directory=output_dir)
        self.logger.info("Saving models checkpoint to %s", output_dir)

    def _save_best_checkpoint(self, best_step):
        model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(self.args.output_dir, f"checkpoint-{best_step}"),
        )
        model.save_pretrained(self.args.output_dir)
        torch.save(
            self.args, os.path.join(self.args.output_dir, "training_args.bin")
        )
        self.tokenizer.save_vocabulary(save_directory=self.args.output_dir)
        self.logger.info("Saving models checkpoint to %s", self.args.output_dir)
