from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    """The interface every agent (DQN, PPO, A2C) implements.

    `mask` is a bool array over N_ACTIONS marking legal actions at the
    current node (see zsrl.env.legal_actions). It must be respected in
    act(), stored alongside the next state in observe() (state vectors
    alone can't be masked again later), and applied when valuing the next
    state inside update() -- masking after the fact, rather than penalising
    illegal actions, is the whole point: the agent should never spend
    training signal on learning the tree's own rules.
    """

    @abstractmethod
    def act(self, state: np.ndarray, mask: np.ndarray, greedy: bool = False) -> int:
        """Pick a legal action for `state`. `greedy=True` disables exploration."""
        raise NotImplementedError

    @abstractmethod
    def observe(self, s: np.ndarray, a: int, r: float, s2: np.ndarray,
                done: bool, next_mask: np.ndarray) -> None:
        """Record one transition, including the mask for the *next* state."""
        raise NotImplementedError

    @abstractmethod
    def update(self) -> float | None:
        """One learning step, if enough data has been observed. Returns the
        loss, or None if no update happened this call."""
        raise NotImplementedError

    @abstractmethod
    def q_values(self, state: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Per-action values for `state`. If `mask` is given, illegal actions
        must not be able to win an argmax over the result (e.g. set to -inf)."""
        raise NotImplementedError
