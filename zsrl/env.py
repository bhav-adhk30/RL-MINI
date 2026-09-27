from collections import deque

import numpy as np
from PIL import Image

from zsrl import CANVAS, MAX_DEPTH, MIN_COMMIT_DEPTH

TL, TR, BL, BR, UP, COMMIT = 0, 1, 2, 3, 4, 5
N_ACTIONS = 6
ACTION_NAMES = ["top-left", "top-right", "bottom-left", "bottom-right",
                "back up", "commit"]


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
    if depth > 0:
        mask[UP] = True
    if depth >= min_commit:
        mask[COMMIT] = True
    return mask


def random_flip_rotate(img, box, rng, canvas=CANVAS, p=0.75):
    """With probability p, apply one horizontal flip, vertical flip, or
    90/180/270 degree rotation to both the image and its box, chosen
    uniformly. The quadtree partitions the canvas into equal quadrants, so
    it is symmetric under these transforms: this re-presents the same real
    image and the same real defect in a different quadrant, rather than
    synthesising new data. Returns (img, box, tag), where tag identifies
    which transform was applied (or "id" for none), for the encoder cache
    key -- augmented views of the same source image are different crops at
    a given node and must not collide in the cache.
    """
    if rng.random() >= p:
        return img, box, "id"
    x1, y1, x2, y2 = box
    choice = int(rng.integers(5))
    if choice == 0:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        box = (canvas - x2, y1, canvas - x1, y2)
        tag = "flipx"
    elif choice == 1:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        box = (x1, canvas - y2, x2, canvas - y1)
        tag = "flipy"
    elif choice == 2:
        img = img.transpose(Image.ROTATE_90)
        box = (y1, canvas - x2, y2, canvas - x1)
        tag = "rot90"
    elif choice == 3:
        img = img.transpose(Image.ROTATE_180)
        box = (canvas - x2, canvas - y2, canvas - x1, canvas - y1)
        tag = "rot180"
    else:
        img = img.transpose(Image.ROTATE_270)
        box = (canvas - y2, x1, canvas - y1, x2)
        tag = "rot270"
    return img, box, tag


class ZoomSearchEnv:
    def __init__(self, canvases, encoder, budget=12, lam=0.1,
                 history_len=4, corruption=None, corruption_tag="clean",
                 augment=False, seed=0):
        self.canvases = canvases
        self.encoder = encoder
        self.budget = budget
        self.lam = lam
        self.history_len = history_len
        self.corruption = corruption
        self.corruption_tag = corruption_tag
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.state_dim = encoder.dim + (MAX_DEPTH + 1) + 4 + 1 + history_len * N_ACTIONS

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
        self.t = 0
        self.visited = [self.node]
        self.visited_nodes = {self.node}
        return self._state()

    def _state(self):
        feat = self.encoder.encode_node(self.image, self.image_key, self.node)

        depth_oh = np.zeros(MAX_DEPTH + 1, dtype=np.float32)
        depth_oh[self.node[0]] = 1.0

        x1, y1, x2, y2 = node_box(self.node)
        pos = np.array([x1 / CANVAS, y1 / CANVAS,
                        (x2 - x1) / CANVAS, (y2 - y1) / CANVAS], dtype=np.float32)

        left = np.array([(self.budget - self.t) / self.budget], dtype=np.float32)

        hist = np.zeros(self.history_len * N_ACTIONS, dtype=np.float32)
        for i, a in enumerate(self.history):
            hist[i * N_ACTIONS + a] = 1.0

        return np.concatenate([feat, depth_oh, pos, left, hist]).astype(np.float32)

    def legal(self):
        return legal_actions(self.node)

    def step(self, action):
        self.t += 1

        if action == COMMIT:
            ok = (contains_centre(self.node, self.target)
                  and self.node[0] >= MIN_COMMIT_DEPTH)
            # Wrong commit is -1, not -3: with a guaranteed -3 for timing
            # out (below), E[commit] = 3p - 1(1-p) = 4p - 1 is positive for
            # any p > 0.25, comfortably inside what a partly-trained agent
            # can reach, rather than needing p > 0.5 to be worth attempting
            # at all. See results/reward_shaping_notes.md.
            reward = 3.0 if ok else -1.0
            info = {"success": bool(ok), "committed": True, "steps": self.t,
                    "depth": self.node[0], "node": self.node}
            return self._state(), reward, True, info

        if action == UP:
            self.node = parent(self.node)
            reward = -0.2
        else:
            self.node = child(self.node, action)
            first_visit = self.node not in self.visited_nodes
            self.visited_nodes.add(self.node)
            if first_visit:
                reward = (1.0 if contains_centre(self.node, self.target) else -1.0) - self.lam
            else:
                # A down-up-down cycle nets +0.7 under the always-paid
                # descent reward (+0.9 descend, -0.2 up, +0.9 redescend),
                # which is trivially farmable for return without ever
                # localising. Paying the +-1 component only once per node,
                # per episode removes the incentive to cycle while leaving
                # the first descent into every node exactly as shaped as
                # before -- a practical, potential-based fix, not a
                # reward-function redesign.
                reward = -self.lam

        self.history.appendleft(action)
        self.visited.append(self.node)

        done = self.t >= self.budget
        if done:
            # No free option to give up: timing out without committing pays
            # -3, at least as bad as a wrong commit. Under the old 0-reward
            # timeout, E[commit] = 3p - 3(1-p) = 6p - 3 was negative for any
            # localisation accuracy below 50%, so a partly-trained agent
            # could rationally learn to always run out the clock instead of
            # ever attempting a commit -- and once committing stops
            # happening, it can never learn to commit correctly either.
            # See results/reward_shaping_notes.md.
            reward -= 3.0
        info = {"success": False, "committed": False, "steps": self.t,
                "depth": self.node[0], "node": self.node}
        return self._state(), reward, done, info
