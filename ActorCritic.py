import numpy as np

from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam

class ActorCritic(object):
    def __init__(self, alpha, beta, gamma=0.99, n_actions=4, layer1_size=1024,
                 layer2_size=512, input_dims=8):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        self.action_space = [i for i in range(self.n_actions)]

    def build_actor_critic_network(self):
        input = Input(shape=(self.input_dims,))
        delta = Input(shape=[1])
        dense1 = Dense(self.fc1_dims, activation='relu')(input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        probs = Dense(self.n_actions, activation='softmax')(dense2)
        values = Dense(1, activation='linear')(dense2)

        def loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*delta)

        actor = Model(input=[input, delta], output=[probs])
        actor.compile(optimizer=Adam(lr=self.alpha), loss=loss)

        critic = Model(input=[input], output=[values])
        critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')

        policy = Model(input=[input], output=[probs])

        return actor, critic, policy

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)

        return action

    def learn(self, state, action, reward, state_, done):
        state = state[np.newaxis, :]
        state_ = state_[np.newaxis, :]

        cr