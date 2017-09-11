import tensorflow as tf
import numpy as np

from params import *
from agent import Agent
from model import Model

import gym

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    env = gym.make(ENVIRONMENT)

    model = Model(env)
    q1 = model.create_model()
    q2 = model.create_model()

    agent = Agent(q1, q2)

    for e in range(EPISODES):
        state = np.expand_dims(env.reset(), axis=0)

        for t in range(200):
            env.render()

            action = agent.act(state, env.action_space)

            state_new, reward, done, _ = env.step(action)
            state_new = np.expand_dims(state_new, axis=0)

            agent.remember(state, action, reward, state_new, done)

            state = state_new

            if done:
                print('hi')

                break
