import tensorflow as tf
import numpy as np

import random

from params import *
from collections import deque


class Agent:
    """ Network's intelligent agent.

    Attributes:
        q1: A model of the first q network.
        q2: A model of the second q network.
    """

    def __init__(self, q1, q2):
        """Initialize Agent.

        Args:
            q1: A model of the first q network.
            q2: A model of the second q network.
        """
        self.q1 = q1
        self.q2 = q2
        self.memory = deque(maxlen=MAX_MEMORY_CAPACITY)

    def remember(self, state, action, reward, new_state, done):
        """Adds data to memory.

        Memory contains (state, action, reward, new_state, done).

        Args:
            state: A numpy array representing the original state of the environment.
            action: A numpy int64 representing the action the agent sends to the environment.
            reward: A float representing the reward the agent receives from the action.
            new_state: A numpy array representing the state of the environment after the action.
            done: A boolean representing if the environment finished a game after the action.
        """
        self.memory.append((state, action, reward, new_state, done))

    def act(self, state, action_space):
        """Select the next action.

        Determine which action the agent should send to the environment based on the state. There is an epsilon chance
        that it explores by selecting a random action.

        Args:
            state: A numpy array representing that state of the environment.
            action_space: A discrete space containing the possible action the agent can give to the environment.

        Returns:
            An integer in the action space (0 or 1).
        """
        if np.random.rand() <= EPSILON:
            # Select a random action
            action = action_space.sample()
        else:
            q1_out = self.q1.predict(state)
            q2_out = self.q2.predict(state)

            action = np.argmax((q1_out + q2_out)[0])

        return action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, new_state, done in minibatch:
            if np.random.rand() <= 0.5:
                target = reward

                if not done:
                    pass
            else:
                pass
