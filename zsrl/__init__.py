import os

import torch

CANVAS = 1024          # canvas side in pixels
MAX_DEPTH = 4          # deepest level, region is 64px
MIN_COMMIT_DEPTH = 3   # cannot commit above this


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def output_root():
    """Kaggle persists /kaggle/working/ as the notebook's saved output;
    anywhere else (including a plain relative path, if the session's cwd
    isn't /kaggle/working) is lost the moment the session ends. Elsewhere
    (local dev), use the current directory, matching the existing
    runs/<agent>/seed<N>_lam<L>/ convention."""
    return "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
