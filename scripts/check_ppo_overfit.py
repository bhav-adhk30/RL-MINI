"""Step 4 gate for zsrl/agents/ppo.py: can PPO overfit a single training
image? Passes when return reaches ~5.7 (the optimal descent-then-commit
return) within roughly 400 episodes, committing successfully in ~4 steps.
If it can't, the environment is wrong -- see docs/IMPLEMENTATION.md's
troubleshooting table. Do not tune hyperparameters instead of debugging.

Note: rollout_len is shrunk to 256 here (vs. 2048 for real training) so
PPO actually gets several updates within a single-image gate's short
episodes. This is a gate-script setting only -- full training runs use
the spec hyperparameters (rollout_len=2048).
"""
from zsrl import get_device
from zsrl.agents.ppo import PPOAgent
from zsrl.dataset import MVTecDefects
from zsrl.encoder import FrozenEncoder
from zsrl.env import N_ACTIONS, ZoomSearchEnv

encoder = FrozenEncoder(get_device())
env = ZoomSearchEnv(MVTecDefects(split="train"), encoder, seed=0)
agent = PPOAgent(env.state_dim, N_ACTIONS, get_device(), rollout_len=256)

for ep in range(400):
    s = env.reset(index=0)
    total = 0.0
    while True:
        a = agent.act(s, env.legal())
        s2, r, done, info = env.step(a)
        agent.observe(s, a, r, s2, done, env.legal())
        agent.update()
        s, total = s2, total + r
        if done:
            break
    if ep % 50 == 0:
        print(ep, "return", round(total, 2), "success", info["success"],
              "steps", info["steps"])
