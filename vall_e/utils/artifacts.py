from collections import defaultdict

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.markers import MarkerStyle

from .distributed import is_local_leader
from .trainer import get_cfg, get_iteration


def is_saving():
    try:
        cfg = get_cfg()
        itr = get_iteration()
    except:
        return False

    return (
        is_local_leader()
        and cfg is not None
        and cfg.save_artifacts_every is not None
        and itr is not None
        and (itr % cfg.save_artifacts_every == 0)
    )


def get_cfg_itr_strict():
    cfg = get_cfg()
    itr = get_iteration()
    assert cfg is not None
    assert itr is not None
    return cfg, itr


def get_path(name, suffix, mkdir=True):
    cfg, itr = get_cfg_itr_strict()
    path = (cfg.log_dir / "artifacts" / name / f"{itr:06d}").with_suffix(suffix)
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_fig(name):
    path = get_path(name, ".png")
    plt.savefig(path)
    plt.close()
    print(path, "saved.")


def save_wav(name, wav, sr):
    # Lazy import
    import soundfile

    path = get_path(name, ".wav")
    soundfile.write(str(path), wav, sr)
    print(path, "saved.")


def save_tsne(
    name,
    x: np.ndarray | list,
    y: np.ndarray | list | None = None,
    c: np.ndarray | list | None = None,
    n_jobs: int = 8,
):
    """
    Args:
        x: list of vectors.
        y: list of labels.
        c: list of colors.
    """
    # Lazy import
    from openTSNE import TSNE

    x = np.array(x)

    if y is not None:
        y = list(y)

    if c is not None:
        c = list(cm.rainbow(np.array(c)))

    x = TSNE(n_components=2, n_jobs=n_jobs).fit(x)

    groups = defaultdict(list)

    z = [None] * len(x)

    for xi, yi, ci in zip(x, y or z, c or z):
        groups[yi].append((*xi, ci))

    for (ki, vi), mi in zip(sorted(groups.items()), MarkerStyle.markers):
        ai, bi, ci = zip(*vi)
        if any([cij is None for cij in ci]):
            # Only use different markers when color is given.
            ci = None
            mi = None
        plt.scatter(x=ai, y=bi, c=ci, alpha=0.5, label=str(ki), marker=mi)

    plt.legend()

    save_fig(name)
