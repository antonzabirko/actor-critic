import numpy as np

from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam

class ActorCritic(object):
    def __init__(self, alpha, beta, gamma=0.99, n_actions=2, layer1_size=1024,
                 layer2_size=512, input_dims=4):
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

        actor = Model(inputs=[input, delta], outputs=[probs])
        actor.compile(optimizer=Adam(lr=self.alpha), loss='mean_squared_error')

        critic = Model(inputs=[input], outputs=[values])
        critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')

        policy = Model(inputs=[input], outputs=[probs])

        return actor, critic, policy

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)

        return action

    def learn(self, state, action, reward, state_, done):
        state = state[np.newaxis, :]
        state_ = state_[np.newaxis, :]

        critic_value_ = self.critic.predict(state_)
        critic_value = self.critic.predict(state)

        target = reward + self.gamma*critic_value_*(1-int(done))
        delta = target - critic_value

        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1.0

        actor_history = self.actor.fit([state, delta], actions, verbose=0)
        self.critic.fit(state, target, verbose=0)

        return actor_history