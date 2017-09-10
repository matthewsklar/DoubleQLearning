import tensorflow as tf

from params import *

import gym

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    env = gym.make(ENVIRONMENT)
