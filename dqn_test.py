import gymnasium as gym
import random
import numpy as np
from collections import deque
from IPython import display
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

def rgb2gray(img):
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    Gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    Gray = np.reshape(Gray, (210, 160, 1))
    return Gray.astype(np.uint8)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95   
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(210, 160, 1)))
        model.add(Conv2D(16, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(self.action_size, activation='relu'))
        model.compile(loss='mean_squared_error', optimizer='sgd')
        return model

    def memorize(self, state, action, reward, next_state, done):
        state = rgb2gray(state)
        next_state = rgb2gray(next_state)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = rgb2gray(state)
        state = np.expand_dims(state, axis=0)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        
EPISODES = 20
visualize = True

if __name__ == "__main__":
    env = gym.make('Breakout-v0', render_mode='rgb_array')
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    #agent.load("./save/dqn_conv.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        for time in range(500):
            if visualize:
                screen = env.render()
                images = [Image.fromarray(screen)]
                display.clear_output(wait=True)
                display.display(plt.gcf())
                plt.imshow(screen)
            else:
                env.render()
            if time != 0:
                action = agent.act(state)
            else:
                action = 0
            next_state, reward, done, _ = env.step(action)[:4]
            reward = reward if not done else -10
            if time != 0:
                agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size+1:
                agent.replay(batch_size)
                
            if time % 10 == 0 and visualize:
                images.append(Image.fromarray(screen))
                
        if e % 10 == 0:
            agent.save("./save/dqn_conv.h5")
    agent.save("./save/dqn_conv.h5")