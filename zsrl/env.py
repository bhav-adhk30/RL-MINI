"""Quadtree search environment, v2.

Changes from v1 (tag env-frozen-v1), each justified by a measurement in
results/FINAL_RESULTS.md -- see docs/REBUILD_V2.md:

1. UP removed. Episodes that used it scored 0.162 recall at 10.05 steps
   against 0.467 at 4.03 for episodes that did not; corr(steps, success) was
   -0.377. Disabling it on unchanged v1 weights raised recall 0.299 -> 0.321
   and halved the cost. It was a symptom of being lost, not a recovery.
2. COMMIT legal from depth 1 instead of depth 3. With UP gone and a forced
   depth-3 commit the task degenerates into three independent 4-way
   classifications, which is not an RL problem. Choosing when to stop is.
3. Commit reward scales with depth, so the policy must trade localisation
   precision against the risk of descending into the wrong quadrant.
4. Cost is measured in 224-equivalent encoder units: a depth-0 node is encoded
   at 448 and costs 4, every deeper node costs 1 (see zsrl/encoder.py).
"""
from collections import deque

import numpy as np
from PIL import Image

from zsrl import CANVAS, MAX_DEPTH, MIN_COMMIT_DEPTH

TL, TR, BL, BR, COMMIT = 0, 1, 2, 3, 4
N_ACTIONS = 5
ACTION_NAMES = ["top-left", "top-right", "bottom-left", "bottom-right", "commit"]


def units(depth):
    """224-equivalent compute to encode a node at `depth`.

    Depth 0 is encoded at 448x448 because a 60px defect occupies 13px of a 224
    input there and 26px of a 448 one; the linear probe at depth 0 goes 0.522
    -> 0.701 with that change. A 448 pass is 4x the FLOPs of a 224 pass, so it
    is counted as 4 units. Reporting raw pass counts would silently make every
    pass four times more expensive and the headline metric dishonest.
    """
    return 4 if depth == 0 else 1


def node_box(node, canvas=CANVAS):
    depth, row, col = node
    size = canvas // (2 ** depth)
    return (col * size, row * size, (col + 1) * size, (row + 1) * size)


def child(node, quadrant):
    depth, row, col = node
    return (depth + 1, 2 * row + quadrant // 2, 2 * col + quadrant % 2)


def parent(node):
    depth, row, col = node
    return (depth - 1, row // 2, col // 2)


def contains_centre(node, box, canvas=CANVAS):
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    x1, y1, x2, y2 = node_box(node, canvas)
    return x1 <= cx < x2 and y1 <= cy < y2


def legal_actions(node, max_depth=MAX_DEPTH, min_commit=MIN_COMMIT_DEPTH):
    depth = node[0]
    mask = np.zeros(N_ACTIONS, dtype=bool)
    if depth < max_depth:
        mask[TL:BR + 1] = True
    if depth >= min_commit:
        mask[COMMIT] = True
    return mask


def random_flip_rotate(img, box, rng, canvas=CANVAS, p=0.75):
    """With probability p apply one flip or 90/180/270 rotation to image and
    box together. The quadtree partitions the canvas symmetrically under these
    transforms, so this re-presents the same real defect in a different
    quadrant rather than synthesising data. `tag` keys the encoder cache."""
    if rng.random() >= p:
        return img, box, "id"
    x1, y1, x2, y2 = box
    choice = int(rng.integers(5))
    if choice == 0:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        box = (canvas - x2, y1, canvas - x1, y2); tag = "flipx"
    elif choice == 1:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        box = (x1, canvas - y2, x2, canvas - y1); tag = "flipy"
    elif choice == 2:
        img = img.transpose(Image.ROTATE_90)
        box = (y1, canvas - x2, y2, canvas - x1); tag = "rot90"
    elif choice == 3:
        img = img.transpose(Image.ROTATE_180)
        box = (canvas - x2, canvas - y2, canvas - x1, canvas - y1); tag = "rot180"
    else:
        img = img.transpose(Image.ROTATE_270)
        box = (canvas - y2, x1, canvas - y1, x2); tag = "rot270"
    return img, box, tag


class ZoomSearchEnv:
    """budget is in 224-equivalent units, not actions.

    Every node the agent OBSERVES costs units(depth): the root costs 4 (it is
    encoded at 448), every deeper node costs 1. A descent to depth 3 therefore
    costs 4 + 1 + 1 + 1 = 7, and the deepest reachable node (depth 4) costs 8,
    which is the default budget.

    Note the agent pays one unit more than a fixed-depth descent would: it has
    to observe the node it is standing on in order to decide whether to commit
    there. A supervised classifier with a hardcoded stopping depth never pays
    that unit, and also cannot choose when to stop.
    """
    def __init__(self, canvases, encoder, budget=8, lam=0.1, base=2.0,
                 history_len=4, corruption=None, corruption_tag="clean",
                 augment=False, seed=0):
        self.canvases = canvases
        self.encoder = encoder
        self.budget = budget
        self.lam = lam
        self.base = base
        self.history_len = history_len
        self.corruption = corruption
        self.corruption_tag = corruption_tag
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.state_dim = (encoder.dim + (MAX_DEPTH + 1) + 4 + 1
                          + history_len * N_ACTIONS)

    def commit_reward(self, depth):
        """Correct commit pays more the deeper it is, so one more descent has
        to be worth the risk of entering the wrong quadrant. `base` selects
        the operating point on the measured precision/recall curve
        (0.530 @ 512px, 0.425 @ 256px, 0.321 @ 128px)."""
        return float(self.base ** (depth - 1))

    def reset(self, index=None):
        if index is None:
            index = int(self.rng.integers(len(self.canvases)))
        path, box, label, key = self.canvases[index]

        img = Image.open(path).convert("RGB").resize((CANVAS, CANVAS))
        if self.corruption is not None:
            img = self.corruption(img)

        aug_tag = "id"
        if self.augment:
            img, box, aug_tag = random_flip_rotate(img, box, self.rng)

        self.image = img
        self.image_key = key + "|" + self.corruption_tag + "|" + aug_tag
        self.target = box
        self.label = label
        self.node = (0, 0, 0)
        self.history = deque(maxlen=self.history_len)
        self.t = 0                    # actions taken
        self.spent = units(0)         # 224-equivalent units: the root is observed
        self.visited = [self.node]
        return self._state()

    def _state(self):
        feat = self.encoder.encode_node(self.image, self.image_key, self.node)

        depth_oh = np.zeros(MAX_DEPTH + 1, dtype=np.float32)
        depth_oh[self.node[0]] = 1.0

        x1, y1, x2, y2 = node_box(self.node)
        pos = np.array([x1 / CANVAS, y1 / CANVAS,
                        (x2 - x1) / CANVAS, (y2 - y1) / CANVAS], dtype=np.float32)

        left = np.array([(self.budget - self.spent) / self.budget], dtype=np.float32)

        hist = np.zeros(self.history_len * N_ACTIONS, dtype=np.float32)
        for i, a in enumerate(self.history):
            hist[i * N_ACTIONS + a] = 1.0

        return np.concatenate([feat, depth_oh, pos, left, hist]).astype(np.float32)

    def legal(self):
        mask = legal_actions(self.node)
        # A descent that cannot be paid for is not a legal action. Same masking
        # mechanism as the tree's own rules, new reason -- carried into the
        # term project, where affordability masking does the same job.
        if self.spent + units(self.node[0] + 1) > self.budget:
            mask[TL:BR + 1] = False
        if not mask.any():
            mask[COMMIT] = True          # never leave the agent with no move
        return mask

    def step(self, action):
        self.t += 1

        if action == COMMIT:
            ok = (contains_centre(self.node, self.target)
                  and self.node[0] >= MIN_COMMIT_DEPTH)
            reward = self.commit_reward(self.node[0]) if ok else -1.0
            info = {"success": bool(ok), "committed": True, "steps": self.t,
                    "units": self.spent, "depth": self.node[0], "node": self.node}
            return self._state(), reward, True, info

        # a descent pays for observing the node it arrives at
        self.node = child(self.node, action)
        cost = units(self.node[0])
        self.spent += cost
        reward = (1.0 if contains_centre(self.node, self.target) else -1.0) \
            - self.lam * cost

        self.history.appendleft(action)
        self.visited.append(self.node)

        # Episodes end only on COMMIT. After any descent the node is at depth
        # >= 1, so COMMIT is always legal and the agent is never stranded; the
        # budget mask caps how deep it can go. Worst case is MAX_DEPTH descents
        # followed by a commit, so episodes are bounded at MAX_DEPTH + 1 steps.
        info = {"success": False, "committed": False, "steps": self.t,
                "units": self.spent, "depth": self.node[0], "node": self.node}
        return self._state(), reward, False, info
