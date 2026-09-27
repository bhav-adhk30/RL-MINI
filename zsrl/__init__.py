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
