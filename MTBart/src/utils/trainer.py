import json

from transformers.utils import logging
import sys

sys.path.append(".")
from transformers import Trainer,set_seed
from data.utils import move_to_cuda
import torch
from typing import Optional, Union, Dict, Any, Tuple, List
import optuna
import collections
import math
from transformers.trainer_callback import TrainerState
import time
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers.modeling_utils import PreTrainedModel
import torch.distributed as dist
from transformers.trainer_utils import speed_metrics, TrainOutput
from transformers.file_utils import WEIGHTS_NAME
import os
from dataclasses import dataclass
import torch.nn as nn
from transformers.trainer_pt_utils import nested_detach, DistributedLengthGroupedSampler, LengthGroupedSampler
from transformers.trainer_utils import PredictionOutput
import datasets
from packaging import version
from data.dataset import SkipDistributedSampler
from torch.utils.data.sampler import RandomSampler
from transformers.training_args import ParallelMode

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

logger = logging.get_logger(__name__)
question_column_name = "question"
context_column_name = "context"
answer_column_name = "answers"
ID = "id"


def emptyCollateFunc(batch):
    return batch


@dataclass
class CustomLabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (:obj:`float`, `optional`, defaults to 0.1):
            The label smoothing factor.
        ignore_index (:obj:`int`, `optional`, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    def __init__(self, epsilon=0.1, ignore_index=-100):
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def __call__(self, model_output, labels, padding_mask=None):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        if padding_mask is None:
            padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = labels.clamp(0)
        if logits.dim() < 3:
            log_probs *= ~padding_mask

        nll_loss = log_probs.gather(dim=-1, index=labels)
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True)

        if logits.dim() >= 3:
            nll_loss.masked_fill_(padding_mask, 0.0)
            smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        if logits.dim() >= 3:
            nll_loss = nll_loss.sum() / num_active_elements
            smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        else:
            nll_loss = nll_loss.sum() / logits.size(0)
            smoothed_loss = smoothed_loss.sum() / (num_active_elements)

        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


class MTGenerationTrainer(Trainer):
    """
    Generated Trainer for Tud.
    """

    def __init__(self, decoders=['L2R', 'R2L', 'BERT'], *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.label_smoothing_factor > 0:
            self.label_smoother = CustomLabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None
        self.use_amp = False
        self.use_apex = False
        self.deepspeed = False
        self.decoders = decoders
        self.losses = {decoder: 0.0 for decoder in self.decoders}

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        inputs = move_to_cuda(inputs)
        #print(inputs)
        labels = {}

        for decoderType in self.decoders:
            labels[decoderType] = inputs['labels'].pop(decoderType)

        for decoderType in ['L2R', 'R2L', 'BERT']:
            if decoderType in inputs:
                inputs['labels'].pop(decoderType)

        outputs = model(**inputs)

        loss = 0.0
        part_losses = {decoder: 0.0 for decoder in self.decoders}
        vocab_size = len(self.tokenizer)
        if self.label_smoother is None:
            loss_fn = torch.nn.CrossEntropyLoss()
            for decoder_type in outputs['logits']:
                cur_loss = loss_fn(outputs['logits'][decoder_type].view(-1, vocab_size), labels[decoder_type].view(-1))
                part_losses[decoder_type] += cur_loss
                loss += cur_loss
        else:
            for decoder_type in outputs['logits']:
                cur_loss = self.label_smoother((outputs['logits'][decoder_type].view(-1, vocab_size),),
                                               labels[decoder_type].view(-1))
                part_losses[decoder_type] += cur_loss
                loss += cur_loss

        losses = [loss]
        losses += [part_losses[decoder] for decoder in self.decoders]
        #return torch.stack(losses)
        return torch.tensor(losses,requires_grad=True)


    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # compute acc and ppl
        total = {decoderType: 0.0 for decoderType in self.decoders}
        right = {decoderType: 0.0 for decoderType in self.decoders}
        losses = {decoderType: 0.0 for decoderType in self.decoders}

        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        print("\n****** Evaluate ******\n")

        self.model.eval()
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = move_to_cuda(batch)
                labels = {}

                for decoderType in self.decoders:
                    labels[decoderType] = batch.pop(decoderType)

                outputs = self.model(**batch)
                for decoderType in self.decoders:
                    logits = outputs['logits'][decoderType]
                    pred = torch.argmax(logits, dim=-1)
                    mask = torch.ne(labels[decoderType], -100).type(torch.long)
                    total[decoderType] += torch.sum(mask).item()
                    right[decoderType] += torch.sum(torch.eq(labels[decoderType], pred).type(torch.int) * mask).item()
                    losses[decoderType] += loss_fn(logits.view(-1, logits.size(-1)),
                                                   labels[decoderType].view(-1)).item()

            evalResult = {}
            for decoderType in self.decoders:
                evalResult[f"{decoderType}-ACC"] = format(right[decoderType] / total[decoderType], '.4f')
                evalResult[f"{decoderType}-PPL"] = format(math.exp(losses[decoderType] / total[decoderType]), '.2f')

        print(json.dumps(evalResult, indent=4))
        self.model.train()

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]):
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        total = loss[0]

        if self.use_amp:
            self.scaler.scale(total).backward()
        elif self.use_apex:
            with self.amp.scale_loss(total, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(total)
        else:
            total.backward()
        return loss.detach()

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)

            for decoder in self.decoders:
                logs[decoder] = round(self.losses[decoder] / (self.state.global_step - self._globalstep_last_logged), 4)
                self.losses[decoder] = 0  # reset

            # backward compatibility for pytorch schedulers
            logs["learning_rate"] = (
                self.lr_scheduler.get_last_lr()[0]
                if version.parse(torch.__version__) >= version.parse("1.4")
                else self.lr_scheduler.get_lr()[0]
            )
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    # def _get_skip_train_sampler(self, skip=0):
    #     if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
    #         self.train_dataset, collections.abc.Sized
    #     ):
    #         return None
    #
    #     # Gather the number of processes and this process index.
    #     if self.args.parallel_mode == ParallelMode.TPU:
    #         num_processes = xm.xrt_world_size()
    #         process_index = xm.get_ordinal()
    #     elif (
    #         self.args.parallel_mode == ParallelMode.DISTRIBUTED
    #         or self.args.parallel_mode == ParallelMode.SAGEMAKER_DISTRIBUTED
    #     ):
    #         num_processes = dist.get_world_size()
    #         process_index = dist.get_rank()
    #     else:
    #         num_processes = 1
    #         process_index = 0
    #
    #     # Build the sampler.
    #     if self.args.group_by_length:
    #         if num_processes <= 1:
    #             return LengthGroupedSampler(self.train_dataset, self.args.train_batch_size)
    #         else:
    #             return DistributedLengthGroupedSampler(
    #                 self.train_dataset, self.args.train_batch_size, num_replicas=num_processes, rank=process_index
    #             )
    #
    #     else:
    #         if num_processes <= 1:
    #             return RandomSampler(self.train_dataset)
    #         else:
    #             return SkipDistributedSampler(self.train_dataset, num_replicas=num_processes, rank=process_index,
    #                                           skip=skip)
    #
    # def get_skip_train_dataloader(self, skip=0):
    #     if self.train_dataset is None:
    #         raise ValueError("Trainer: training requires a train_dataset.")
    #     train_sampler = self._get_skip_train_sampler(skip)
    #
    #     return DataLoader(
    #         self.train_dataset,
    #         batch_size=self.args.train_batch_size,
    #         sampler=train_sampler,
    #         collate_fn=self.data_collator,
    #         drop_last=self.args.dataloader_drop_last,
    #         num_workers=self.args.dataloader_num_workers,
    #         pin_memory=self.args.dataloader_pin_memory,
    #     )

    def get_updated_train_dataloader(self, skipNum=0):
        self.train_dataset.setSkipNum(skipNum)
        return self.get_train_dataloader()

    def train(
        self,
        resume_from_checkpoint: Optional[str] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (:obj:`str`, `optional`):
                Local path to a saved checkpoint as saved by a previous instance of :class:`~transformers.Trainer`. If
                present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            '''warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )'''
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
            print(f"Loading model from {resume_from_checkpoint}).")
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(resume_from_checkpoint)
                model_reloaded = True
            else:
                state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if not self.is_model_parallel:
                self.model = self.model.to(self.args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        if train_dataset_is_sized:
            if self.args.local_rank != -1:
                world_size = dist.get_world_size()
            else:
                world_size = 1
            num_update_steps_per_epoch = len(self.train_dataset) // (
                self.args.gradient_accumulation_steps * self.args.per_device_train_batch_size * world_size)
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                    self.args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, "trainer_state.json")
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not self.args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            print("  Continuing training from checkpoint, will skip to saved global_step")
            print(f"  Continuing training from epoch {epochs_trained}")
            print(f"  Continuing training from global step {self.state.global_step}")
            if not self.args.ignore_data_skip:
                print(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch."
                )

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        model = self.model_wrapped

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.deepspeed:
            pass  # already initialized its own DDP earlier
        elif self.args.local_rank != -1:
            if self.args.ddp_find_unused_parameters is not None:
                find_unused_parameters = self.args.ddp_find_unused_parameters
            elif isinstance(model, PreTrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                find_unused_parameters = not getattr(model.config, "gradient_checkpointing", False)
            else:
                find_unused_parameters = True
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=find_unused_parameters,
            )

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), DDP(Deepspeed(Transformers Model)), etc.

        # Train!
        if self.args.local_rank != -1:
            world_size = dist.get_world_size()
        else:
            world_size = 1

        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps * world_size
        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * self.args.max_steps
        )

        print("***** Running training *****")
        print(f"  Num examples = {num_examples}")
        print(f"  Num Epochs = {num_train_epochs}")
        print(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        print(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {max_steps}")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        #self.state.trial_params = hp_params(trial) if trial is not None else None
        self.state.trial_params = self.hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(self.args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        self._total_flos = self.state.total_flos
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not self.args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break
        #??
        #train_dataloader = self.get_updated_train_dataloader(
          #  skipNum=steps_trained_in_current_epoch * self.args.train_batch_size // self.args.dataloader_num_workers - self.args.train_batch_size)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if train_dataset_is_sized
                else self.args.max_steps * self.args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_in_current_epoch % 10000 == 0:
                        print(f"left {steps_trained_in_current_epoch}")
                    continue
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                if ((step + 1) % self.args.gradient_accumulation_steps != 0) and self.args.local_rank != -1:
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        losses = self.training_step(model, inputs)
                        tr_loss += losses[0]
                else:
                    losses = self.training_step(model, inputs)
                    tr_loss += losses[0]

                for idx, decoder in enumerate(self.decoders):
                    self.losses[decoder] += losses[idx + 1].item()

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= self.args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.use_amp:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                self.args.max_grad_norm,
                            )

                            # amp??

                    # Optimizer step
                    if self.deepspeed:
                        self.deepspeed.step()
                    elif self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

            if self.args.tpu_metrics_debug or self.args.debug:
                logger.warning(
                    "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                    "configured. Check your training configuration if this is unexpected."
                )
            if self.control.should_training_stop:
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        print("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            print(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(self.state.best_model_checkpoint)
                if not self.is_model_parallel:
                    self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        metrics = speed_metrics("train", start_time, self.state.max_steps)
        if self._total_flos is not None:
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
        self.log(metrics)

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()

        return TrainOutput(self.state.global_step, self._total_loss_scalar / self.state.global_step, metrics)


class NoNumelTrainer(Trainer):
    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (:obj:`str`, `optional`):
                Local path to a saved checkpoint as saved by a previous instance of :class:`~transformers.Trainer`. If
                present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
            print(f"Loading model from {resume_from_checkpoint}).")
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(resume_from_checkpoint)
                model_reloaded = True
            else:
                state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if not self.is_model_parallel:
                self.model = self.model.to(self.args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                    self.args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        model = self.model_wrapped

        # Mixed precision training with apex (torch < 1.6)
        if self.use_apex:
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.deepspeed:
            pass  # already initialized its own DDP earlier
        elif self.args.local_rank != -1:
            if self.args.ddp_find_unused_parameters is not None:
                find_unused_parameters = self.args.ddp_find_unused_parameters
            elif isinstance(model, PreTrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                find_unused_parameters = not getattr(model.config, "gradient_checkpointing", False)
            else:
                find_unused_parameters = True
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=find_unused_parameters,
            )

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), DDP(Deepspeed(Transformers Model)), etc.

        # Train!
        if self.args.local_rank != -1:
            world_size = dist.get_world_size()
        else:
            world_size = 1

        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps * world_size
        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * self.args.max_steps
        )

        print("***** Running training *****")
        print(f"  Num examples = {num_examples}")
        print(f"  Num Epochs = {num_train_epochs}")
        print(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        print(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, "trainer_state.json")
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not self.args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            print("  Continuing training from checkpoint, will skip to saved global_step")
            print(f"  Continuing training from epoch {epochs_trained}")
            print(f"  Continuing training from global step {self.state.global_step}")
            if not self.args.ignore_data_skip:
                print(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = self.hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(self.args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        self._total_flos = self.state.total_flos
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not self.args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if train_dataset_is_sized
                else self.args.max_steps * self.args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                if ((step + 1) % self.args.gradient_accumulation_steps != 0) and self.args.local_rank != -1:
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss += self.training_step(model, inputs)
                else:
                    tr_loss += self.training_step(model, inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= self.args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.use_amp:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                self.args.max_grad_norm,
                            )

                    # Optimizer step
                    if self.deepspeed:
                        self.deepspeed.step()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        print("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            print(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(self.state.best_model_checkpoint)
                if not self.is_model_parallel:
                    self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        metrics = speed_metrics("train", start_time, self.state.max_steps)
        if self._total_flos is not None:
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
        self.log(metrics)

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()

        return TrainOutput(self.state.global_step, self._total_loss_scalar / self.state.global_step, metrics)


# TODO: Sequence Classification Trainer
class MTSequenceClassificationTrainer(MTGenerationTrainer):
    def __init__(self, is_regression=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_regression = is_regression
        self.label_smoother = None

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss_pre(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]

            else:
                loss = None
                if self.use_amp:
                    with autocast():
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def compute_loss_pre(self, model, inputs, return_outputs=False):
        assert "labels" in inputs

        inputs = move_to_cuda(inputs)

        if not self.is_regression:
            # FIXME: label smoother
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            loss_fn = torch.nn.MSELoss()

        labels = inputs.pop("labels")
        outputs = model(**inputs)

        if not return_outputs:
            return loss_fn(outputs['logits'], labels)
        else:
            return loss_fn(outputs['logits'], labels), outputs


class QuestionAnsweringTrainer(NoNumelTrainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

        if self.args.label_smoothing_factor != 0:
            self.label_smoother = CustomLabelSmoother(epsilon=self.args.label_smoothing_factor)

    def compute_loss(self, model, inputs, return_outputs=False):
        start_labels = inputs.pop("start_positions")
        end_labels = inputs.pop("end_positions")

        inputs = move_to_cuda(inputs)

        outputs = model(**inputs)
        start_logits = outputs[0]
        end_logits = outputs[1]

        if self.label_smoother is not None:
            loss_fn = self.label_smoother
            padding_mask = inputs['decoder_input_ids']['BERT'].eq(self.tokenizer.pad_token_id)
            loss = loss_fn((start_logits,), start_labels, padding_mask) + loss_fn((end_logits,), end_labels,
                                                                                  padding_mask)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(start_logits, start_labels) + loss_fn(end_logits, end_labels)
        return loss

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        # We might have removed columns from the dataset so we put them back.
        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(type=eval_dataset.format["type"], columns=list(eval_dataset.features.keys()))

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, test_dataset, test_examples, ignore_keys=None):
        test_dataloader = self.get_test_dataloader(test_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                test_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        # We might have removed columns from the dataset so we put them back.
        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(type=test_dataset.format["type"], columns=list(test_dataset.features.keys()))

        eval_preds = self.post_process_function(test_examples, test_dataset, output.predictions)
        metrics = self.compute_metrics(eval_preds)

        return PredictionOutput(predictions=eval_preds.predictions, label_ids=eval_preds.label_ids, metrics=metrics)

    def _move_to_device(self, inputs):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)
            elif isinstance(v, dict):
                inputs[k] = self._move_to_device(v)

        return inputs

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._move_to_device(inputs)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                if self.use_amp:
                    with autocast():
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        return (loss, logits, labels)


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


class BartQuestionAnsweringTrainer(QuestionAnsweringTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        start_labels = inputs.pop("start_positions")
        end_labels = inputs.pop("end_positions")

        inputs = move_to_cuda(inputs)

        outputs = model(**inputs)
        start_logits = outputs[0]
        end_logits = outputs[1]

        if self.label_smoother is not None:
            loss_fn = self.label_smoother
            padding_mask = shift_tokens_right(inputs['input_ids'], self.tokenizer.pad_token_id).eq(
                self.tokenizer.pad_token_id)
            loss = loss_fn((start_logits,), start_labels, padding_mask) + loss_fn((end_logits,), end_labels,
                                                                                  padding_mask)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(start_logits, start_labels) + loss_fn(end_logits, end_labels)
        return loss
