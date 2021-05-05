from base import Agent
import numpy as np
from collections import defaultdict
from utils.policies import RandomPolicy, GreedyPolicy

__all__ = ['ValueIter', 'PolicyIter']


class DpBase(Agent):
    """
    Base class for dp algorithms, assume environment is given, in other words transition probability
    and reward distribution is given, state space and action space have to be discrete.

    """

    def __init__(self, env, threshold=10e-3, gamma=0.99, val_func='Q', max_eval_iter=500, max_ctl_iter=100):

        super().__init__(env)
        self.threshold = threshold
        self.gamma = gamma
        self.val_func = val_func
        self.max_eval_iter = max_eval_iter
        self.max_ctl_iter = max_ctl_iter
        self._val = defaultdict(lambda: [0] * env.action_space.n)

    def train(self):

        raise NotImplementedError

    def evaluate(self, pi):

        dynamics = self.env.env.P
        a_n = self.env.action_space.n
        i = 0

        while i < self.max_eval_iter:

            delta = 0

            for s in dynamics.keys():
                prev_val = self._val[s].copy()
                curr_val = self._val[s]

                for a in dynamics[s].keys():
                    temp_val = 0

                    for p, s_prime, r, _ in dynamics[s][a]:
                        v_next = 0

                        for a_prime in range(a_n):
                            v_next = v_next + self._val[s_prime][a_prime] * pi[s_prime][a_prime]

                        temp_val = temp_val + p * (r + self.gamma * v_next)

                    curr_val[a] = temp_val

                curr_delta = np.linalg.norm(np.array(prev_val) - np.array(curr_val))

                if curr_delta > delta:
                    delta = curr_delta

            if delta < self.threshold:
                break
            i += 1

        return self._val

    def predict(self, s):

        pi_g = GreedyPolicy(self._val)

        return np.argmax(pi_g[s])

    def sample(self):

        pass


class ValueIter(DpBase):

    def train(self):

        dynamics = self.env.env.P
        i = 0

        while i < self.max_ctl_iter:

            delta = 0
            # Calculate Q^*, V^* (s) = max_a Q^* (s, a)
            for s in dynamics.keys():
                prev_val = self._val[s].copy()
                curr_val = self._val[s]

                for a in dynamics[s].keys():
                    temp_val = 0

                    for p, s_prime, r, _ in dynamics[s][a]:
                        temp_val = temp_val + p * (r + self.gamma * np.max(self._val[s_prime]))

                    curr_val[a] = temp_val

                curr_delta = np.linalg.norm(np.array(prev_val) - np.array(curr_val))

                if curr_delta > delta:
                    delta = curr_delta

            if delta < self.threshold:
                break

            i += 1


class PolicyIter(DpBase):

    def train(self):

        n_act = self.env.action_space.n
        i = 0
        old_policy = RandomPolicy(n_act)

        while i < self.max_ctl_iter:
            print(i)
            policy_stable = True
            curr_policy = GreedyPolicy(self._val)

            for s in self.env.env.P.keys():
                old_action = np.random.choice(range(n_act), p=old_policy[s])
                g_action = np.random.choice(range(n_act), p=curr_policy[s])

                if old_action != g_action:
                    policy_stable = False

            if not policy_stable:
                _ = self.evaluate(curr_policy)

            else:
                break

            i += 1
