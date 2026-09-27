"""Standalone sanity check for zsrl/env.py, ahead of baselines.py / the Step 4 gate.

Checks:
  1. state_dim is 546 (512 feature + 5 depth one-hot + 4 pos + 1 budget + 24 history).
  2. legal_actions masks the right actions at every depth.
  3. A hand-driven correct descent (mirroring baselines.oracle) reaches the
     target and returns ~5.7 in 4 steps, matching the guide's reference optimum.
  4. A deliberately wrong first descent gets the wrong-quadrant penalty.
"""
from zsrl import get_device
from zsrl.dataset import MVTecDefects
from zsrl.encoder import FrozenEncoder
from zsrl.env import (BR, COMMIT, TL, UP, N_ACTIONS, ZoomSearchEnv,
                       child, contains_centre)

train = MVTecDefects(split="train")
encoder = FrozenEncoder(get_device())
env = ZoomSearchEnv(train, encoder, seed=0)

print("state_dim", env.state_dim, "expected", 512 + 5 + 4 + 1 + 4 * N_ACTIONS)

s = env.reset(index=0)
print("state shape matches state_dim:", s.shape[0] == env.state_dim)

for depth, mask in [(0, env.legal())]:
    print(f"depth {depth} legal mask", mask.tolist(),
          "(expect descends True, up False, commit False)")

# drive to depth 3 by manually stepping up/down to inspect masks along the way
node = env.node
for _ in range(4):
    node = child(node, 0)
env.node = node
print("depth 4 legal mask", env.legal().tolist(),
      "(expect descends False, up True, commit True)")

# ---- correct oracle-style rollout ----------------------------------------
s = env.reset(index=0)
target = env.target
total = 0.0
node = (0, 0, 0)
trace = []
for _ in range(3):
    for q in range(4):
        c = child(node, q)
        if contains_centre(c, target):
            s2, r, done, info = env.step(q)
            trace.append((q, r, info["node"]))
            total += r
            node = info["node"]
            break
s2, r, done, info = env.step(COMMIT)
trace.append(("commit", r, info))
total += r

print("\ncorrect descent trace:")
for t in trace:
    print(" ", t)
print("total return", round(total, 3), "(expect ~5.7)")
print("success", info["success"], "steps", info["steps"], "(expect True, 4)")

# ---- deliberately wrong first descent -------------------------------------
s = env.reset(index=0)
target = env.target
wrong_q = next(q for q in range(4) if not contains_centre(child((0, 0, 0), q), target))
s2, r, done, info = env.step(wrong_q)
print("\nwrong first descent reward", r, "(expect -1.1)")
