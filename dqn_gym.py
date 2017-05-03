import gym
import numpy as np
import collections
import random
from keras.layers import Dense, LSTM
from keras import layers
from keras.models import Sequential

class DQN:
    def __init__(self):
        self.batch_size = 32
        self.memsize = 1000 # experience replay
        self.state_size = 4
        self.action_size = 2
        self.dis_factor = .9

        # Store experience replace in a ring buffer
        self.d = collections.deque(maxlen=self.memsize)

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def update_target_model (self):
        self.target_model.set_weights( self.model.get_weights() )

    def store_mem (self, obs, action, reward, next_obs, done):
        self.d.append( (obs, action, reward, next_obs, done) )

    def get_action (self, obs):
        return np.argmax( self.model.predict(obs)[0] )
        #return 1 if self.model.predict(obs)[0][0] > .5 else 0

    def train_replay (self):
        # Sample from replay memory
        batchsize = min(self.batch_size, len(self.d))
        minibatch = random.sample(self.d, batchsize)

        input_batch = np.zeros((batchsize, self.state_size))
        target_batch = np.zeros((batchsize, self.action_size))

        for i in range(batchsize):
            obs, action, reward, next_obs, done = minibatch[i]

            # Make all gradients effectively 0
            #target = self.model.predict(obs.reshape(1,obs.size))
            target = self.model.predict(obs.reshape(1,obs.size))[0]

            # Reassign target for non-zero action gradient
            if done:
                target[action] = reward
            else:
                future = np.amax(
                        self.target_model.predict(next_obs.reshape(1,next_obs.size))[0] )
                target[action] = reward + self.dis_factor * future

            # Construct training data
            target_batch[i] = target
            input_batch[i]  = obs

        self.model.fit(input_batch, target_batch, batch_size=batchsize,
                epochs=1, verbose=0)

    def save_model (self, name):
        self.model.save_weights(name)

    def load_model (self, name):
        self.model.load_weights(name)

    def build_model (self):
        model = Sequential()
        model.add( Dense(24, input_shape=(self.state_size,), activation='relu') )
        model.add( Dense(24, activation='relu') )
        model.add( Dense(self.action_size) )

        model.compile(loss='mse', optimizer='adam')
        print( model.summary() )

        return model

if __name__ == '__main__':
    # Init agent
    agent = DQN()

    # Init env
    env = gym.make('CartPole-v1')

    # Parameters
    epsilon = .99
    epsilon_end = 0.005
    epsilon_decay = .0005
    memsize = 1000 # experience replay
    num_epochs = 1000

    # Accumulated reward
    score = 0
    #agent.load_model('dqn.h5')

    for epoch in range(num_epochs):
        obs = env.reset()
        done = False

        # Store previous observation
        prev_obs = obs
        score = 0

        agent.save_model('dqn.h5')

        while not done:
            '''
            env.render()
            action = agent.get_action(obs.reshape(1,obs.size))
            obs, r, done, info = env.step(action)
            score += r
            '''

            if epoch % 10 == 0:
                env.render()

            if np.random.rand() <= epsilon: # Random choice to perform DQN action
                action = env.action_space.sample()
            else:
                action = agent.get_action(obs.reshape(1,obs.size))

            # Perform an action, get an observation
            obs, r, done, info = env.step(action)
            r = r if not done or r == 499 else -100
            agent.store_mem(prev_obs, action, r, obs, done)

            if (epsilon > epsilon_end):
                epsilon -= epsilon_decay

            agent.train_replay()

            prev_obs = obs

            score += r # Accumulate reward

            if done:
                agent.update_target_model()

        print('epoch [', epoch, '] - ', score, ' | e: ', epsilon)
