from base import PolicyBase
import numpy as np
from utils.value_functions import ActionValFunc, StateValFunc
from collections import defaultdict


class GreedyPolicy(PolicyBase):
    """

    Greedy Policy class, given a value function Q or V, return a probability distribution

    \pi(a | s) s.t probability of argmax_a val_func(a) = 1, otherwise = 0, for all a, s


    """

    def __init__(self, val, env=None, gamma=None):
        """
        init GreedyPolicy \pi_g

        :param val: the value function
        :param env:
        :param gamma:
        """
        self._val = None
        self.env = env
        self.gamma = gamma
        self.val = val

    def _get_policy(self, s):

        if not self.env and not self.gamma:
            self._val = self.val[s]

        else:
            dynamics = self.env.env.P
            self._val = [0] * self.env.action_space.n

            for a in dynamics[s].keys():

                for prob, s_prime, r, _ in dynamics[s][a]:
                    self.val[a] = self.val[a] + prob * (r + self.gamma * self.val[s_prime])

    def __getitem__(self, s):

        self._get_policy(s)
        actions = np.arange(len(self._val))
        best_action = np.random.choice(actions[self._val == np.max(self._val)])
        A = np.where(actions == best_action, 1.0, 0.0)

        return A


class Policy(PolicyBase):

    def __init__(self, policy):

        self.policy = policy

    def __getitem__(self, s):

        if callable(self.policy):

            return self.policy(s)

        else:

            return self.policy[s]


class RandomPolicy(PolicyBase):

    def __init__(self, n_act):

        self._policy = defaultdict(lambda: [1 / n_act] * n_act)

    def __getitem__(self, s):

        return self._policy[s]