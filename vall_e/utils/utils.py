import logging
import re
from logging import StreamHandler
from pathlib import Path
from typing import Callable, TypeVar, overload

import pandas as pd
from coloredlogs import ColoredFormatter
from torch import Tensor, nn

from .distributed import global_rank, local_rank

T = TypeVar("T")


def flatten_dict(d):
    records = pd.json_normalize(d).to_dict(orient="records")
    return records[0] if records else {}


def _get_named_modules(module, attrname):
    for name, module in module.named_modules():
        if hasattr(module, attrname):
            yield name, module


def gather_attribute(module, attrname, delete=True, prefix=True):
    ret = {}
    for name, module in _get_named_modules(module, attrname):
        ret[name] = getattr(module, attrname)
        if delete:
            try:
                delattr(module, attrname)
            except Exception as e:
                raise RuntimeError(f"{name} {module} {attrname}") from e
    if prefix:
        ret = {attrname: ret}
    ret = flatten_dict(ret)
    # remove consecutive dots
    ret = {re.sub(r"\.+", ".", k): v for k, v in ret.items()}
    return ret


def dispatch_attribute(
    module,
    attrname,
    value,
    filter_fn: Callable[[nn.Module], bool] | None = None,
):
    for _, module in _get_named_modules(module, attrname):
        if filter_fn is None or filter_fn(module):
            setattr(module, attrname, value)


def load_state_dict_non_strict(model, state_dict, logger=None):
    model_state_dict = model.state_dict()
    provided = set(state_dict)
    required = set(model_state_dict)
    agreed = provided & required
    for k in list(agreed):
        if model_state_dict[k].shape != state_dict[k].shape:
            agreed.remove(k)
            provided.remove(k)
    state_dict = {k: state_dict[k] for k in agreed}
    if logger is not None and (diff := provided - required):
        logger.warning(
            f"Extra parameters are found. "
            f"Provided but not required parameters: \n{diff}."
        )
    if logger is not None and (diff := required - provided):
        logger.warning(
            f"Some parameters are missing. "
            f"Required but not provided parameters: \n{diff}."
        )
    model.load_state_dict(state_dict, strict=False)


def setup_logging(log_dir: str | Path | None = "log", log_level="info"):
    handlers = []
    stdout_handler = StreamHandler()
    stdout_handler.setLevel(logging.INFO)
    formatter = ColoredFormatter(
        f"%(asctime)s - %(name)s - %(levelname)s - GR={global_rank()};LR={local_rank()} - \n%(message)s"
    )
    stdout_handler.setFormatter(formatter)
    handlers.append(stdout_handler)
    if log_dir is not None:
        filename = Path(log_dir) / f"log.txt"
        filename.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(filename, mode="a")
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)
    logging.basicConfig(
        level=logging.getLevelName(log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - \n%(message)s",
        handlers=handlers,
    )


@overload
def tree_map(fn: Callable, x: list[T]) -> list[T]:
    ...


@overload
def tree_map(fn: Callable, x: tuple[T]) -> tuple[T]:
    ...


@overload
def tree_map(fn: Callable, x: dict[str, T]) -> dict[str, T]:
    ...


@overload
def tree_map(fn: Callable, x: T) -> T:
    ...


def tree_map(fn: Callable, x):
    if isinstance(x, list):
        x = [tree_map(fn, xi) for xi in x]
    elif isinstance(x, tuple):
        x = (tree_map(fn, xi) for xi in x)
    elif isinstance(x, dict):
        x = {k: tree_map(fn, v) for k, v in x.items()}
    elif isinstance(x, Tensor):
        x = fn(x)
    return x


def to_device(x: T, device) -> T:
    return tree_map(lambda t: t.to(device), x)
