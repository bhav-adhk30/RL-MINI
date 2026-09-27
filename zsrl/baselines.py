import numpy as np
from PIL import Image

from zsrl import MAX_DEPTH, MIN_COMMIT_DEPTH
from zsrl.env import child, contains_centre, node_box


def oracle(target, depth=MIN_COMMIT_DEPTH):
    """Always descends correctly. Cost = depth. This is the ceiling."""
    node = (0, 0, 0)
    for _ in range(depth):
        for q in range(4):
            c = child(node, q)
            if contains_centre(c, target):
                node = c
                break
    return {"success": True, "cost": depth, "node": node}


def random_descent(target, rng, budget=12, depth=MIN_COMMIT_DEPTH):
    """Random quadrant each step. The floor."""
    node, cost = (0, 0, 0), 0
    while node[0] < depth and cost < budget:
        node = child(node, int(rng.integers(4)))
        cost += 1
    return {"success": contains_centre(node, target) and node[0] >= depth,
            "cost": cost, "node": node}


def odd_one_out_scores(image, node):
    """Per-quadrant anomaly score: squared distance from a quadrant's
    downsampled grayscale appearance to the mean of its three siblings.
    On MVTec, the guide's original 'highest pixel variance' heuristic scores
    0.052 recall (below the guide's expected 0.3-0.6), because busy but
    defect-free texture (screw threads, weave, grain) has more raw variance
    than a subtle scratch. Measured against siblings instead of in absolute
    terms, a quadrant that breaks a repeating pattern stands out even when
    it isn't the highest-variance one; this brings recall to 0.134."""
    crops = []
    for q in range(4):
        c = child(node, q)
        crop = np.asarray(
            image.crop(node_box(c)).resize((32, 32)).convert("L"),
            dtype=np.float32)
        crops.append(crop)
    scores = []
    for q in range(4):
        siblings = [crops[j] for j in range(4) if j != q]
        sib_mean = np.mean(siblings, axis=0)
        scores.append(float(np.mean((crops[q] - sib_mean) ** 2)))
    return scores


def saliency_descent(image, target, budget=12, depth=MIN_COMMIT_DEPTH):
    """Descend into the child that looks most unlike its three siblings
    (see odd_one_out_scores). A sensible engineer's solution, and a
    genuinely tough opponent -- though on MVTec a meaningfully weaker one
    than the guide's original VOC-calibrated expectations, see
    docs/IMPLEMENTATION.md."""
    node, cost = (0, 0, 0), 0
    while node[0] < depth and cost + 4 <= budget:
        scores = odd_one_out_scores(image, node)
        cost += 4
        node = child(node, int(np.argmax(scores)))
    return {"success": contains_centre(node, target) and node[0] >= depth,
            "cost": cost, "node": node}


def normal_model_descent(image, target, reference, encode_fn,
                          budget=12, depth=MIN_COMMIT_DEPTH):
    """Descend by distance from a per-category 'normal' reference.

    `reference` is a (4, encoder.dim) array of mean frozen-encoder features
    for the four depth-1 quadrants, built only from defect-free train/good/
    images of the same category (see scripts/build_normal_reference.py).
    `encode_fn` maps a PIL crop to a feature vector with the same encoder.

    A reference this way exists only at depth 1 (building one at depths 2
    and 3 needs ~20x more train/good images encoded -- hours on a CPU
    kernel, for a classical baseline). So only the first, most informative
    descent uses the learned-normal-model distance; depths 2 and 3 fall
    back to odd_one_out_scores.
    """
    node, cost = (0, 0, 0), 0
    if reference is not None and node[0] < depth and cost + 4 <= budget:
        feats = [encode_fn(image.crop(node_box(child(node, q))))
                 for q in range(4)]
        cost += 4
        dists = [float(np.linalg.norm(feats[q] - reference[q])) for q in range(4)]
        node = child(node, int(np.argmax(dists)))
    while node[0] < depth and cost + 4 <= budget:
        scores = odd_one_out_scores(image, node)
        cost += 4
        node = child(node, int(np.argmax(scores)))
    return {"success": contains_centre(node, target) and node[0] >= depth,
            "cost": cost, "node": node}


def exhaustive(target, depth=MIN_COMMIT_DEPTH):
    """Visit every node at the target depth. Always succeeds, always expensive."""
    return {"success": True, "cost": 4 ** depth, "node": None}
