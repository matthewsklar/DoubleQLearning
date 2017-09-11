import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense


class Model:
    """The model of the neural network.

    The model of a Double Deep Q Learning network.

    Attributes:
        env: A TimeLimit gym wrapper storing the environment
    """
    def __init__(self, env):
        """

        :param env:
        """
        self.env = env

    def create_model(self):
        """ Create the networks model.

        Create the model of the Double Deep Q Learning network.

        Return:
            A Sequential Keras model
        """
        model = Sequential()
        model.add(Dense(128, input_shape=self.env.observation_space.shape, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(optimizer='adam', loss='mse')

        return model
