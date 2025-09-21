from abc import ABC, abstractmethod

class TBuffer(ABC):
    """
    Abstract base class for experience buffers used in reinforcement learning.

    Buffers are used to store experiences (state, action, reward, next_state, done)
    during training and allow sampling or clearing of those experiences.

    Subclasses must implement the methods for remembering experiences,
    clearing the buffer, and returning its length.
    """
    @abstractmethod
    def Remember(self, Date : tuple) -> None:
        """
        Add a new experience to the buffer.

        :param Date: The experience data, typically a tuple of (state, action, reward, next_state, done).
        :type Date: tuple
        """
        pass

    @abstractmethod
    def Clear(self) -> None:
        """
        Clear the buffer of all stored experiences.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the current number of experiences in the buffer.

        :return: Size of the buffer.
        :rtype: int
        """
        pass