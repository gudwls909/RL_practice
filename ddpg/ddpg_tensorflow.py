import cv2
import os
import sys
import gym
import random
import numpy as np
import argparse
import tensorflow as tf
from collections import deque
import sys
import time


class Environment(object):
    def __init__(self, env, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        pass

    def render_worker(self, render):
        if render:
            self.env.render()
        pass

    def new_episode(self):
        state = self.env.reset()
        #state = np.reshape(state, [1, self.state_size])
        return state
        pass

    def act(self, action):
        next_state, reward, terminal, _ = self.env.step(action)
        return next_state, reward, terminal
        pass


class ReplayMemory(object):
    def __init__(self, env, state_size, batch_size):
        self.memory = deque(maxlen=10000)
        self.env = env
        self.state_size = state_size
        self.batch_size = batch_size
        pass

    def add(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))
        pass

    def mini_batch(self):
        mini_batch = random.sample(self.memory, self.batch_size)  # memory에서 random하게 sample

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


class DDPG(object):
    def __init__(self, state_size,  action_size, sess, learning_rate_actor, learning_rate_critic,
                 replay, discount_factor, a_bound):
        self.state_size = state_size
        self.action_size = action_size
        self.sess = sess
        self.lr_actor = learning_rate_actor
        self.lr_critic = learning_rate_critic
        self.replay = replay
        self.discount_factor = discount_factor
        self.action_limit = a_bound

        self.state = tf.placeholder(tf.float32, [None, self.state_size])
        self.target = tf.placeholder(tf.float32, [None, 1])

        self.actor = self.build_actor('actor_eval', True)
        self.actor_target = self.build_actor('actor_target', False)
        self.critic = self.build_critic('critic_eval', True, self.actor)
        self.critic_target = self.build_critic('critic_target', False, self.actor_target)

        self.actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_eval')
        self.actor_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_target')
        self.critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_eval')
        self.critic_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_target')

        self.replace = [tf.assign(t, (1 - 0.01) * t + 0.01 * e)
                   for t, e in zip(self.actor_target_vars + self.critic_target_vars, self.actor_vars + self.critic_vars)]

        self.train_actor = self.actor_optimizer()
        self.train_critic = self.critic_optimizer()
        pass

    def build_actor(self, scope, trainable):
        actor_hidden_size = 30
        with tf.variable_scope(scope):
            hidden1 = tf.layers.dense(self.state, actor_hidden_size, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(hidden1, self.action_size, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.action_limit, name='scaled_a')

    def build_critic(self, scope, trainable, a):
        with tf.variable_scope(scope):
            critic_hidden_size =30
            hidden1 = tf.layers.dense(self.state, critic_hidden_size, name='s1', trainable=trainable) \
                      + tf.layers.dense(a, critic_hidden_size, name='a1', trainable=trainable) \
                      + tf.get_variable('b1', [1, critic_hidden_size], trainable=trainable)
            hidden1 = tf.nn.relu(hidden1)
            return tf.layers.dense(hidden1, 1, trainable=trainable)

    def actor_optimizer(self):
        loss = tf.reduce_mean(self.critic)
        train_op = tf.train.AdamOptimizer(-self.lr_actor).minimize(loss, var_list=self.actor_vars)

        return train_op
        pass

    def critic_optimizer(self):
        loss = tf.losses.mean_squared_error(labels=self.target, predictions=self.critic)
        #loss = tf.reduce_mean(tf.square(self.target - self.critic))
        train_op = tf.train.AdamOptimizer(self.lr_critic).minimize(loss, var_list=self.critic_vars)
        return train_op
        pass

    def train_network(self):
        states, actions, rewards, next_states, terminals = self.replay.mini_batch()

        next_target_q = self.sess.run(self.critic_target, feed_dict={self.state: next_states})

        target = []
        for i in range(self.replay.batch_size):
            if terminals[i]:
                target.append(rewards[i])
            else:
                target.append(rewards[i] + self.discount_factor * next_target_q[i])
        target = np.reshape(target, [self.replay.batch_size, 1])

        self.sess.run(self.train_actor, feed_dict={self.state: states})
        self.sess.run(self.train_critic, feed_dict={self.state: states, self.target: target, self.actor: actions})
        pass

    def update_target_network(self):
        self.sess.run(self.replace)
        pass


class Agent(object):
    def __init__(self, args, sess):
        # CartPole 환경
        self.env = gym.make(args.env_name)
        self.sess = sess
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.env._max_episode_steps = 200 # 최대 타임스텝 수
        self.a_bound = self.env.action_space.high
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.discount_factor = args.discount_factor
        self.episodes = args.episodes
        self.ENV = Environment(self.env, self.state_size, self.action_size)
        self.replay = ReplayMemory(self.env, self.state_size, self.batch_size)
        self.ddpg = DDPG(self.state_size, self.action_size, self.sess, self.learning_rate[0], self.learning_rate[1],
                       self.replay, self.discount_factor, self.a_bound)
        self.saver = tf.train.Saver()
        self.epsilon = 1
        self.explore = 2e4
        pass

    '''
    def select_action(self, state):
        return np.clip(
            np.random.normal(self.sess.run(self.ddpg.actor, {self.ddpg.state: state})[0], self.action_variance), -2,
            2)
        pass
    '''

    def ou_function(self, mu, theta, sigma):
        x = np.ones(self.action_size) * mu
        dx = theta * (mu - x) + sigma * np.random.randn(self.action_size)
        return x + dx

    def noise_select_action(self, state):
        action = self.sess.run(self.ddpg.actor, {self.ddpg.state: state})[0]
        noise = self.epsilon * self.ou_function(0, 0.15, 0.25)
        #print(action, noise)
        return np.clip(action + noise, -2, 2)

    def select_action(self, state):
        return np.clip(self.sess.run(self.ddpg.actor, {self.ddpg.state: state})[0], -2, 2)

    def train(self):
        scores, episodes = [], []
        for e in range(self.episodes):
            terminal = False
            score = 0
            state = self.ENV.new_episode()
            state = np.reshape(state, [1, self.state_size])

            while not terminal:
                #self.epsilon -= 1.0/self.explore
                self.epsilon = max(self.epsilon, 0)
                action = self.noise_select_action(state)
                next_state, reward, terminal = self.ENV.act(action)
                state = state[0]
                self.replay.add(state, action, reward / 10, next_state, terminal)

                if len(self.replay.memory) >= self.batch_size:
                    self.ddpg.update_target_network()
                    self.ddpg.train_network()

                score += reward
                state = np.reshape(next_state, [1, self.state_size])

                if terminal:
                    scores.append(score)
                    episodes.append(e)
                    print('episode:', e+1, ' score:', int(score), 'eps:', self.epsilon, ' last 10 mean score', int(np.mean(scores[-min(10, len(scores)):])))

        pass

    def play(self):
        state = self.ENV.new_episode()
        state = np.reshape(state, [1, self.state_size])
        self.ENV.render_worker(True)

        terminal = False
        score = 0
        while not terminal:
            action = self.select_action(state)
            next_state, reward, terminal = self.ENV.act(action)
            next_state = np.reshape(next_state, [1, self.state_size])
            score += reward
            state = next_state
            self.ENV.render_worker(True)
            time.sleep(0.02)
            if terminal:
                return score
        pass

    def save(self):
        checkpoint_dir = 'save'
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))

    def load(self):
        checkpoint_dir = 'save'
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))


if __name__ == "__main__":

    print(sys.executable)
    # parameter 저장하는 parser
    parser = argparse.ArgumentParser(description="Pendulum")
    parser.add_argument('--env_name', default='Pendulum-v0', type=str)
    parser.add_argument('--learning_rate', default=[0.002, 0.001], type=list)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--discount_factor', default=0.9, type=float)
    parser.add_argument('--episodes', default=100, type=float)
    sys.argv = ['-f']
    args = parser.parse_args()

    config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    config.log_device_placement = False
    config.gpu_options.allow_growth = True

    # 학습 or 테스트
    with tf.Session(config=config) as sess:
        agent = Agent(args, sess)
        sess.run(tf.global_variables_initializer())  # tensorflow graph가 다 만들어지고 난 후에 해야됨
        agent.train()
        agent.save()
        agent.load()
        rewards = []
        for i in range(20):
            r = agent.play()
            rewards.append(int(r))
        mean = np.mean(rewards)
        print(rewards)
        print(mean)


