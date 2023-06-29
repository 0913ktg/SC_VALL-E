import os
import socket
from functools import cache, wraps
from typing import Callable


def get_free_port():
    sock = socket.socket()
    sock.bind(("", 0))
    return sock.getsockname()[1]


@cache
def fix_unset_envs():
    envs = dict(
        RANK="0",
        WORLD_SIZE="1",
        MASTER_ADDR="localhost",
        MASTER_PORT=str(get_free_port()),
        LOCAL_RANK="0",
    )

    for key in envs:
        value = os.getenv(key)
        if value is not None:
            return

    for key, value in envs.items():
        os.environ[key] = value


def local_rank():
    return int(os.getenv("LOCAL_RANK", 0))


def global_rank():
    return int(os.getenv("RANK", 0))


def is_local_leader():
    return local_rank() == 0


def is_global_leader():
    return global_rank() == 0


def local_leader_only(fn=None, *, default=None) -> Callable:
    def wrapper(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if is_local_leader():
                return fn(*args, **kwargs)
            return default

        return wrapped

    if fn is None:
        return wrapper

    return wrapper(fn)


def global_leader_only(fn: Callable | None = None, *, default=None) -> Callable:
    def wrapper(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if is_global_leader():
                return fn(*args, **kwargs)
            return default

        return wrapped

    if fn is None:
        return wrapper

    return wrapper(fn)


def nondistributed(fn):
    @global_leader_only()
    @wraps(fn)
    def wrapped(*args, **kwargs):
        # https://github.com/microsoft/DeepSpeed/blob/b47e25bf95250a863edb2c466200c697e15178fd/deepspeed/utils/distributed.py#L34
        # Deepspeed will check all environ before start distributed.
        # To avoid the start of a distributed task, remove one environ is enough.
        # Here we remove local rank.
        local_rank = os.environ.pop("LOCAL_RANK", "")
        ret = fn(*args, **kwargs)
        os.environ["LOCAL_RANK"] = local_rank
        return ret

    return wrapped
