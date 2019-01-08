# coding: utf-8

env = gym.make('CartPole-v0')
state = env.reset()

num_actions = env.action_space.n
state_shape = env.observation_space.shape
print(num_actions)
print(state_shape)

next_state, reward, terminal, info = env.step(env.action_space.sample())

import random


class Environment(object):
    def __init__(self, env, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        pass
    
    def random_action(self):
        # Return a random action.
        return random.randrange(self.action_size)
        pass
    
    def render_worker(self, render=False):
        # If display in your option is true, do rendering. Otherwise, do not.
        if render:
            self.env.render()
        pass
    
    def new_episode(self):
        # Start a new episode and return the first state of the new episode.
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        return state
        pass
    
    def act(self, action):
        # Perform an action which is given by input argument and return the results of acting.
        next_state, reward, terminal, _ = self.env.step(action)
        return next_state, reward, terminal
        pass


from collections import deque
import random


class ReplayMemory(object):
    def __init__(self, state_size, batch_size):
        self.memory = deque(maxlen=2000)
        self.batch_size = batch_size
        self.state_size = env.observation_space.shape[0]
        pass
    
    def add(self, state, action, reward, next_state, terminal):
        # Add current_state, action, reward, terminal, (next_state which can be added by your choice). 
        self.memory.append((state, action, reward, next_state, terminal))
        pass
    
    def mini_batch(self):
        # Return a mini_batch whose data are selected according to your sampling method. (such as uniform-random sampling in DQN papers)
        mini_batch = random.sample(self.memory, self.batch_size)
        
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, terminals = [], [], []
        
        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            next_states[i] = mini_batch[i][3]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            terminals.append(mini_batch[i][4])
            
        return states, actions, rewards, next_states, terminals            
        pass


class DQN(object):
    def __init__(self, state_size, action_size, learning_rate, replay, batch_size, discount_factor):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.replay = replay
        self.batch_size = batch_size
        self.discount_facter = discount_factor
        self.prediction_Q = self.build_network('pred')
        self.target_Q = self.build_network('target')
        pass
    
    def build_network(self, name):
        # Make your a deep neural network
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            model = tf.keras.Sequential()
            model.add(tf.layers.Dense(25, input_dim=self.state_size, activation='relu', 
                            kernel_initializer='he_uniform'))
            model.add(tf.layers.Dense(25, activation='relu', kernel_initializer='he_uniform'))
            model.add(tf.layers.Dense(25, activation='relu', kernel_initializer='he_uniform'))
            model.add(tf.layers.Dense(self.action_size, activation='relu', 
                            kernel_initializer='he_uniform'))
            model.summary()
            model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
            return model
            pass
    
    def train_network(self, discount_factor):
        states, actions, rewards, next_states, terminals = self.replay.mini_batch()
        
        pred_Q = self.prediction_Q.predict(states)
        tar_Q = self.target_Q.predict(next_states)
        
        for i in range(self.batch_size):
            if terminals[i]:
                pred_Q[i][actions[i]] = rewards[i]
            else:
                pred_Q[i][actions[i]] = rewards[i] + discount_factor*(np.amax(tar_Q[i]))
                
        self.prediction_Q.fit(states, pred_Q, batch_size=self.batch_size, epochs=1, verbose=0)
        pass
    
    def update_target_network(self):
        #self.sess.run(copy_op)
        self.target_Q.set_weights(self.prediction_Q.get_weights())
    
    #def predict_Q(self, ...):
    #    pass


# ### Agent class

# In[6]:


import os # to save and load
import random
class Agent(object):
    def __init__(self, args, mode):
        self.env = gym.make(args.env_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        #self.saver = tf.train.Saver()
        if mode=='train':
            self.epsilon = 1.0
        elif mode=='test':
            self.epsilon = 0.0
        else:
            raise Exception("mode type not supported: {}".format(mode))
        self.epsilon_decay_steps = args.epsilon_decay_steps
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.discount_factor = args.discount_factor
        self.episodes = args.episodes
        self.ENV = Environment(self.env, self.state_size, self.action_size)
        self.replay = ReplayMemory(self.state_size, self.batch_size)
        self.dqn = DQN(self.state_size, self.action_size, self.learning_rate, 
                       self.replay, self.batch_size, self.discount_factor)
        pass
    
    def select_action(self, state):
        # Select an action according ε-greedy. You need to use a random-number generating function and add a library if necessary.
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.dqn.prediction_Q.predict(state)
            return np.argmax(q_value[0])
        pass
    
    def train(self):
        # Train your agent which has the neural nets.
        # Several hyper-parameters are determined by your choice (Options class in the below cell)
        # Keep epsilon-greedy action selection in your mind 
                
        scores, episodes = [], []
        
        for e in range(self.episodes):
            terminal = False
            score = 0
            state = self.ENV.new_episode()
            
            int_e = 0
            while not terminal:
                action = self.select_action(state)
                next_state, reward, terminal = self.ENV.act(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                self.replay.add(state, action, reward, next_state, terminal)
                
                if len(self.replay.memory)>=1000:
                    if self.epsilon > 0.1:
                        self.epsilon -= 0.9/1e4
                    self.dqn.train_network(self.discount_factor)
                    
                score += reward
                state = next_state
                int_e += 1
                
                if terminal:
                    self.dqn.update_target_network()
                    scores.append(score)
                    episodes.append(e)
                    print('episode:', e, ' score:', score, ' epsilon', self.epsilon, 
                          ' last 10 mean score', np.mean(scores[-min(10, len(scores)):]))
                    
                    if np.mean(scores[-min(10, len(scores)):]) > 195:
                        print('Already well trained')
                        return
            
        pass
    
    def play(self, test=False):
        # Test your agent 
        # When performing test, you can show the environment's screen by rendering,
        state = self.ENV.new_episode()
        self.ENV.render_worker(test)
        
        terminal = False
        score = 0
        while not terminal:
            action = self.select_action(state)
            next_state, reward, terminal = self.ENV.act(action)
            next_state = np.reshape(next_state, [1, self.state_size])
            score += reward
            state = next_state

            if terminal:
                return score
        pass
    
    def save(self):
        #checkpoint_dir = 'cartpole'
        #if not os.path.exists(checkpoint_dir):
        #    os.mkdir(checkpoint_dir)
        #self.saver.save(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))
        self.dqn.prediction_Q.save_weights("./save_model/dqn.h5")
        
    def load(self):
        #checkpoint_dir = 'cartpole'
        #self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))
        self.dqn.prediction_Q.load_weights("./save_model/dqn.h5")


# ## 2. Train your agent 
# 
# Now, you train an agent to play CartPole-v0. Options class is the collection of hyper-parameters that you can choice. Usage of Options class is not mandatory.<br>
# The maximum value of total reward which can be aquired from one episode is 200. 
# <font color='red'>**You should show learning status such as the number of observed states and mean/max/min of rewards frequently (for instance, every 100 states).**</font>

# In[7]:


import argparse
import sys
parser = argparse.ArgumentParser(description="CartPole")
parser.add_argument('--env_name', default='CartPole-v0', type=str,
                    help="Environment")
#parser.add_argument('--render', default=False, type=bool)
parser.add_argument('--epsilon_decay_steps', default=1000, type=int,
                    help="how many steps for epsilon to be 0.1")
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--discount_factor', default=0.99, type=float)
parser.add_argument('--episodes', default=300, type=float)
sys.argv = ['-f']
args = parser.parse_args()
print(args)
config = tf.ConfigProto()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config.log_device_placement = False
config.gpu_options.allow_growth = True

#with tf.Session(config=config) as sess:
#myAgent = Agent(args, 'train') # It depends on your class implementation
#myAgent.train()
#myAgent.save()


# In[8]:


#with tf.Session(config=config) as sess:
myAgent = Agent(args, 'train') # It depends on your class implementation
myAgent.train()
myAgent.save()


# ## <a name="play"></a> 3. Test the trained agent ( 15 points )
# 
# Now, we test your agent and calculate an average reward of 20 episodes.
# - 0 <= average reward < 50 : you can get 0 points
# - 50 <= average reward < 100 : you can get 10 points
# - 100 <= average reward < 190 : you can get 35 points
# - 190 <= average reward <= 200 : you can get 50 points

# In[9]:


#config = tf.ConfigProto()
# If you use a GPU, uncomment
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#config.log_device_placement = False
# config.gpu_options.allow_growth = True
#with tf.Session(config=config) as sess:
#args = parser.parse_args() # You set the option of test phase
myAgent = Agent(args, 'test') # It depends on your class implementation
myAgent.load()
rewards = []
for i in range(20):
    r = myAgent.play() # play() returns the reward cumulated in one episode
    rewards.append(r)
mean = np.mean(rewards)
print(rewards)
print(mean)

