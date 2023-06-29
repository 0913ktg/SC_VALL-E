import json
import logging
import random
import selectors
import sys
from functools import cache
from typing import Protocol
import os

import humanize
import numpy as np
import torch
from torch.distributed import broadcast_object_list
from torch.utils.data import DataLoader

from .config import Config
from .distributed import (
    global_leader_only,
    global_rank,
    is_global_leader,
    is_local_leader,
    local_leader_only,
)
from .engines import Engine, Engines, TrainFeeder
from .utils import to_device

_logger = logging.getLogger(__name__)
_engines: Engines
_command: str


def get_global_step():
    try:
        return _engines.global_step
    except:
        return None


def get_cfg():
    try:
        return _engines.cfg
    except:
        raise RuntimeError("Trainer has not been setup. Have you called trainer.train?")


def get_cmd():
    try:
        return _command
    except:
        raise RuntimeError("Trainer has not been setup. Have you called trainer.train?")


get_iteration = get_global_step


class EnginesLoader(Protocol):
    def __call__(self) -> Engines:
        ...


def load_engines(engines: dict[str, Engine], config: Config):
    engines = Engines(engines)
    engines.setup(config)
    
    if engines.cfg.is_style_layer:
        # 이전 체크포인트가 있으면 tag를 지정하고 첫 학습이면 tag지정 안함       
        if engines.cfg.model == 'ar':
            if os.path.exists('/data/vall-e/ckpts/korean/ar/model/style'):   
                engines.load_checkpoint(tag='style')     
            else:
                engines.load_checkpoint()
        elif engines.cfg.model == 'nar':
            if os.path.exists('/data/vall-e/ckpts/korean/nar/model/style'):        
                engines.load_checkpoint(tag='style')  
            else:
                engines.load_checkpoint()        
    else:
        engines.load_checkpoint()
    return engines


class EvalFn(Protocol):
    def __call__(self, *, engines: Engines):
        ...


class Logger(Protocol):
    def __call__(self, *, data: dict):
        ...


@cache
def _get_stdin_selector():
    selector = selectors.DefaultSelector()
    selector.register(fileobj=sys.stdin, events=selectors.EVENT_READ)
    return selector


def _non_blocking_input():
    global _command
    l = [""]
    if is_global_leader():
        s = ""
        selector = _get_stdin_selector()
        events = selector.select(timeout=0)
        for key, _ in events:
            s: str = key.fileobj.readline().strip()
            _logger.info(f'Get stdin "{s}".')
        l[0] = s
    broadcast_object_list(l, src=0)
    _command = l[0]
    return _command


def _make_infinite_epochs(dl):
    while True:
        _logger.info("New epoch starts.")
        yield from dl


@local_leader_only(default=None)
def logger(data):
    return _logger.info(json.dumps(data, indent=2, default=str))


def seed(seed):
    # Set up random seeds, after fork()
    random.seed(seed + global_rank())
    np.random.seed(seed + global_rank())
    torch.manual_seed(seed + global_rank())


def train(
    engines_loader: EnginesLoader,
    train_dl: DataLoader,
    train_feeder: TrainFeeder,
    eval_fn: EvalFn,
    logger: Logger = logger,
    is_style_layer = False
):
    engines = engines_loader()
    cfg = engines.cfg

    if is_local_leader():
        cfg.dump()
        _logger.info(cfg)

    # Setup global engines
    global _engines
    _engines = engines

    events = []

    eval_fn = global_leader_only(eval_fn)

    # Pre-loop command
    command = _non_blocking_input()
    if command in ["eval", "eval_quit"]:
        engines.eval()
        eval_fn(engines=engines)
        engines.train()
    if command in ["quit", "eval_quit"]:
        return

    # Training loop
    print('training loop start')
    
    for batch in _make_infinite_epochs(train_dl):
        if engines.global_step >= cfg.max_iter:
            break
        
        batch = to_device(batch, torch.cuda.current_device())
        
        stats = engines.step(feeder=train_feeder, batch=batch)
    
        elapsed_time = stats.get("elapsed_time", 0)
        logger(data=stats)

        command = _non_blocking_input()

        if "@" in command:
            what, when = command.split("@")
            try:
                events.append((what, int(when)))
                _logger.info(f"Event {command} registered.")
            except Exception as e:
                _logger.error(e)
            command = ""

        # Commands are the current command plus the triggered (i.e. iteration >= trigger point) events
        events = [e for e in events if e[1] >= engines.global_step]
        commands = [command] + [e[0] for e in events if e[1] == engines.global_step]

        for command in commands:
            if command in ["event show", "event"]:
                msg = "Events:\n" + "\n".join(["@".join(map(str, e)) for e in events])
                _logger.info(msg)

            if command == "event clear":
                events.clear()

            if "time" in command:
                target_iter = cfg.max_iter
                if " to " in command:
                    try:
                        target_iter = int(command.split(" to ")[-1])
                    except Exception as e:
                        _logger.error(e)
                remaining_iters = target_iter - engines.global_step + 1
                remaining_time = int(remaining_iters * elapsed_time)
                _logger.info(humanize.precisedelta(remaining_time))

            save_ckpt_every = cfg.save_ckpt_every or cfg.eval_every

            saving_commands = ["save"]

            if cfg.save_on_quit:
                saving_commands.append("quit")

            if engines.global_step % save_ckpt_every == 0 or command in saving_commands:
                # style layer가 존재할 경우 다른 체크포인트 사용
                if is_style_layer:
                    engines.save_checkpoint(tag='style')
                else:
                    engines.save_checkpoint()

            if engines.global_step % cfg.eval_every == 0 or command in ["eval"]:
                engines.eval()
                eval_fn(engines=engines)
                engines.train()

            if command in ["quit"]:
                return
        