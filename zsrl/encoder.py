import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights


class FrozenEncoder:
    # Global average pooling to a single 512-vector discards where in the
    # view something is -- exactly the information needed to tell which
    # quadrant holds the defect. Pooling layer4's 7x7 map to 2x2 instead
    # (adaptive_avg_pool2d) keeps one 512-dim summary per quadrant, so one
    # encoder call on the parent view yields TL/TR/BL/BR features in one
    # pass: 2x2x512 = 2048. See results/spatial_pooling_notes.md.
    dim = 2048

    def __init__(self, device, cache_size=400_000):
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for p in net.parameters():
            p.requires_grad = False
        self.net = net.eval().to(device)
        self.device = device

        self.tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.cache = {}
        self.cache_size = cache_size
        self.calls = 0        # every real forward pass, the compute counter
        self.hits = 0

    @torch.no_grad()
    def encode_pil(self, pil_image):
        x = self.tf(pil_image).unsqueeze(0).to(self.device)
        feat = self._spatial_features(x)
        return feat.squeeze(0).float().cpu().numpy()

    def _spatial_features(self, x):
        """layer4 output (B,512,7,7) -> 2x2 adaptive pool -> (B,2048), laid
        out as four contiguous 512-dim blocks [TL, TR, BL, BR], matching
        node_box/child's row(top/bottom), col(left/right) convention."""
        n = self.net
        x = n.conv1(x); x = n.bn1(x); x = n.relu(x); x = n.maxpool(x)
        x = n.layer1(x); x = n.layer2(x); x = n.layer3(x); x = n.layer4(x)
        if x.device.type == "mps":
            # PyTorch's MPS backend doesn't support adaptive_avg_pool2d
            # with a non-divisible input/output ratio (7 -> 2 here); CPU
            # does, and the tensor is tiny at this point, so do this one
            # op on CPU and move back. CUDA (Kaggle) is unaffected.
            x = F.adaptive_avg_pool2d(x.cpu(), (2, 2)).to(x.device)
        else:
            x = F.adaptive_avg_pool2d(x, (2, 2))       # (B, 512, 2, 2)
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)  # (B, 2048)

    def encode_node(self, pil_image, image_key, node):
        """node is (depth, row, col). Exact key, no quantisation needed."""
        key = (image_key, node)
        cached = self.cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached

        self.calls += 1
        from zsrl.env import node_box
        x1, y1, x2, y2 = node_box(node)
        feat = self.encode_pil(pil_image.crop((x1, y1, x2, y2)))
        if len(self.cache) < self.cache_size:
            self.cache[key] = feat
        return feat

    def reset_counter(self):
        self.calls = 0

    def stats(self):
        total = self.calls + self.hits
        return {"calls": self.calls, "hits": self.hits,
                "hit_rate": self.hits / total if total else 0.0,
                "entries": len(self.cache)}
