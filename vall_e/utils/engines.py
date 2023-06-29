import logging
import time
from typing import Any, Protocol
import pickle


import torch
import torch.distributed
from deepspeed import DeepSpeedEngine
from torch import Tensor

from .config import Config

from .distributed import fix_unset_envs
from .utils import dispatch_attribute, flatten_dict, gather_attribute

Stats = dict[str, float]

_logger = logging.getLogger(__name__)


class Engine(DeepSpeedEngine):
    def __init__(self, *args, **kwargs):
        fix_unset_envs()
        super().__init__(None, *args, **kwargs)
        self._frozen_params = set()

    def freeze(self):
        for p in self.module.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
                self._frozen_params.add(p)

    def unfreeze(self):
        for p in self._frozen_params:
            p.requires_grad_(True)
        self._frozen_params.clear()

    @property
    def global_step(self):
        return self.global_steps

    def gather_attribute(self, *args, **kwargs):
        return gather_attribute(self.module, *args, **kwargs)

    def dispatch_attribute(self, *args, **kwargs):
        return dispatch_attribute(self.module, *args, **kwargs)

    @torch.no_grad()
    def compute_grad_norm(self):
        grads = [
            p.grad.float() / self.grad_scale
            for p in self.parameters()
            if p.grad is not None
        ]
        # torch.autograd.set_detect_anomaly(True)
                
        grad_norm = torch.stack([g.detach().norm() for g in grads]).norm()
        return grad_norm

    @property
    def grad_scale(self):
        return getattr(self.optimizer, "cur_scale", 1)


class TrainFeeder(Protocol):
    def __call__(
        self, *, engines: "Engines", batch: Any, name: str
    ) -> None | tuple[Tensor, Stats]:
        ...


class Engines(dict[str, Engine]):
    def setup(self, cfg: Config):
        self._cfg = cfg
        self._global_step = 0

    @property
    def cfg(self) -> Config:
        return self._cfg

    @property
    def config(self):
        return self._cfg

    @property
    def global_step(self):
        return self._global_step

    def gather_attribute(self, *args, **kwargs):
        ret = {}
        for engine in self.values():
            ret |= engine.gather_attribute(*args, **kwargs)
        return ret

    def dispatch_attribute(self, *args, **kwargs):
        for engine in self.values():
            engine.dispatch_attribute(*args, **kwargs)

    def save_checkpoint(self, tag="default"):
        self.cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
        for name, engine in self.items():
            engine.save_checkpoint(self.cfg.ckpt_dir / name, tag=tag)

    def load_checkpoint(self, tag=None, strict=False):
        for name, engine in self.items():
            engine.load_checkpoint(
                self.cfg.ckpt_dir / name,
                tag=tag,
                load_module_strict=strict,
            )
        self._update_global_step()

    def _update_global_step(self):
        for engine in self.values():
            self._global_step = max(self._global_step, engine.global_step)

    def eval(self):
        for engine in self.values():
            engine.eval()

    def train(self):
        for engine in self.values():
            engine.train()

    def step(self, feeder: TrainFeeder, batch):
        total_elapsed_time = 0

        stats: Any = dict()

        for name, engine in self.items():
            torch.cuda.synchronize()
            start_time = time.time()

            oom = False

            try:
                
                maybe_loss_and_engine_stats = feeder(
                    engines=self, batch=batch, name=name
                )

                if maybe_loss_and_engine_stats is None:
                    # Here we allow skip optimizers. It's useful when, for example,
                    # skipping discriminators in the begining of GAN training.
                    continue

                loss, engine_stats = maybe_loss_and_engine_stats
                engine.backward(loss)
                # For monitoring purpose
                grad_norm = engine.compute_grad_norm()
                engine.step()

                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                total_elapsed_time += elapsed_time

                stats.update(
                    flatten_dict(
                        {
                            name: dict(
                                loss=loss.item(),
                                lr=engine.get_lr()[0],
                                grad_norm=grad_norm.item(),
                                elapsed_time=elapsed_time,
                                engine_step=engine.global_step,
                                **engine_stats,
                            )
                        }
                    ),
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.cfg.save_on_oom:
                    oom = True
                else:
                    raise e

            # except ValueError as e:
            #     print('value error by blocks')
            
            # Do a sync here for OOM check.
            torch.distributed.barrier()

            if oom:
                self.save_checkpoint()
                raise RuntimeError("Out of memory!")

        self._update_global_step()
        stats["elapsed_time"] = total_elapsed_time
        stats["wall_time"] = time.time()
        stats["global_step"] = self.global_step

        return stats
