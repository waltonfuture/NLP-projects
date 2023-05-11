from transformers import TrainingArguments
from dataclasses import dataclass, field
import torch
from transformers.file_utils import cached_property, torch_required
import os
from transformers.utils import logging
from datetime import timedelta

logger = logging.get_logger(__name__)


@dataclass
class FileTrainingArguments(TrainingArguments):
    global_rank: int = field(default=-1, metadata={"help": "global rank"})

    world_size: int = field(default=-1,
                            metadata={"help": "distributed world size"})

    cache_file: str = field(default="file_share_cache",
                            metadata={"help": "sharing cache file"})

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif self.deepspeed:
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            from transformers.integrations import is_deepspeed_available

            if not is_deepspeed_available():
                raise ImportError(
                    "--deepspeed requires deepspeed: `pip install deepspeed`.")
            import deepspeed

            deepspeed.init_distributed()

            # workaround for setups like notebooks where the launcher can't be used,
            # but deepspeed requires a dist env.
            # env LOCAL_RANK could be set manually by the user, or via init_distributed if mpi4py is installed
            self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # if self.global_rank == 0 and self.local_rank == 0 and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            timeout = timedelta(days=1)
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            torch.distributed.init_process_group(
                backend="nccl",
                init_method=f"file://{os.path.abspath(self.output_dir)}/{self.cache_file}",
                rank=self.global_rank,
                world_size=self.world_size,
                timeout=timeout)
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device
