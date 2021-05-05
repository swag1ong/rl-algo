from abc import ABC, abstractmethod


class Agent(ABC):
    """
    The abstract base class for all RL algorithms.

    """

    def __init__(self, env):
        """
        Signature for initialization

        :param env: gym env object
        """
        self.env = env

    @abstractmethod
    def train(self, *args, **kwargs):
        """
        Signature for training

        """

        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """
        Signature for evaluation

        """

        pass

    @abstractmethod
    def sample(self, *args, **kwargs):

        pass

    @abstractmethod
    def predict(self, *args, **kwargs):

        pass


class PolicyBase(ABC):
    """
    A base policy class for all policies
    """

    @abstractmethod
    def __getitem__(self, *args, **kwargs):
        pass


class ValFunc(ABC):
    """
    A base value function class for all value functions
    """

    @abstractmethod
    def __getitem__(self, s):
        pass

    @abstractmethod
    def __setitem__(self, i, v):
        pass
