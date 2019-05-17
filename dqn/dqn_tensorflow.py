import cv2
import os
import sys
import gym
import random
import numpy as np
import argparse
import tensorflow as tf
from collections import deque

EPISODES = 1000


class Environment(object):
	def __init__(self, env, state_size, action_size):
		self.env = env
		self.state_size = state_size
		self.action_size = action_size
		pass

	def random_action(self):
		return random.randrange(self.action_size)
		pass

	def render_worker(self, render):
		if render:
			self.env.render()
		pass

	def new_episode(self):
		state = self.env.reset()
		state = np.reshape(state, [1, self.state_size])
		return state
		pass

	def act(self, action):
		next_state, reward, terminal, _ = self.env.step(action)
		return next_state, reward, terminal
		pass


class ReplayMemory(object):
	def __init__(self, env, state_size, batch_size):
		self.memory = deque(maxlen=2000)
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


class DQN(object):
	def __init__(self, state_size, action_size, sess, learning_rate, replay, discount_factor):
		self.state_size = state_size
		self.action_size = action_size
		self.sess = sess
		self.lr = learning_rate
		self.replay = replay
		self.discount_factor = discount_factor

		self.states = tf.placeholder(tf.float32, [None, self.state_size])
		self.actions = tf.placeholder(tf.int64, [None])
		self.target = tf.placeholder(tf.float32, [None])

		self.prediction_Q = self.build_network('pred')
		self.target_Q = self.build_network('target')
		self.train_op = self.build_optimizer()
		pass

	def build_network(self, name):
		with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
			h1 = tf.layers.dense(self.states, 25, activation=tf.nn.relu,
			                     kernel_initializer=tf.initializers.truncated_normal)
			h2 = tf.layers.dense(h1, 25, activation=tf.nn.relu,
			                     kernel_initializer=tf.initializers.truncated_normal)
			output = tf.layers.dense(h2, self.action_size,
			                     kernel_initializer=tf.initializers.truncated_normal)
			return output
			pass

	def build_optimizer(self):
		actions_one_hot = tf.one_hot(self.actions, self.action_size, 1.0, 0.0)
		q_value = tf.reduce_sum(tf.multiply(actions_one_hot, self.prediction_Q), axis=1)
		loss = tf.reduce_mean(tf.square(self.target - q_value))
		train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
		return loss, train_op
		pass

	def train_network(self):
		states, actions, rewards, next_states, terminals = self.replay.mini_batch()

		target_Q = self.sess.run(self.target_Q, feed_dict={self.states: next_states})
		target = []
		for i in range(self.replay.batch_size):
			if terminals[i]:
				target.append(rewards[i])
			else:
				target.append(rewards[i] + self.discount_factor * np.max(target_Q[i]))

		self.sess.run(self.train_op, feed_dict={self.states: states, self.actions: actions, self.target: target})
		pass

	def update_target_network(self):
		copy_op = []
		pred_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pred')
		target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')
		for pred_var, target_var in zip(pred_vars, target_vars):
			copy_op.append(target_var.assign(pred_var.value()))
		self.sess.run(copy_op)

	def predict_Q(self, states):
		return self.sess.run(self.prediction_Q, feed_dict={self.states: states})
		pass


class Agent(object):
	def __init__(self, args, sess):
		# CartPole 환경
		self.env = gym.make(args.env_name)
		self.eps = 1.0  # epsilon
		self.sess = sess
		self.state_size = self.env.observation_space.shape[0]
		self.action_size = self.env.action_space.n
		self.env._max_episode_steps = 10000  # 최대 타임스텝 수 10000
		self.epsilon_decay_steps = args.epsilon_decay_steps
		self.learning_rate = args.learning_rate
		self.batch_size = args.batch_size
		self.discount_factor = args.discount_factor
		self.episodes = args.episodes
		self.ENV = Environment(self.env, self.state_size, self.action_size)
		self.replay = ReplayMemory(self.env, self.state_size, self.batch_size)
		self.dqn = DQN(self.state_size, self.action_size, self.sess, self.learning_rate,
		               self.replay, self.discount_factor)
		self.saver = tf.train.Saver()
		pass

	def select_action(self, state):
		if np.random.rand() <= self.eps:
			return random.randrange(self.action_size)
		else:
			q_value = self.dqn.predict_Q(state)
			return np.argmax(q_value[0])
		pass

	def train(self):
		scores, episodes = [], []
		self.dqn.update_target_network()  # 첫 시작때 target network를 prediction network와 똑같이

		for e in range(self.episodes):
			terminal = False
			score = 0
			state = self.ENV.new_episode()

			while not terminal:
				action = self.select_action(state)
				next_state, reward, terminal = self.ENV.act(action)
				next_state = np.reshape(next_state, [1, self.state_size])
				self.replay.add(state, action, reward, next_state, terminal)

				if len(self.replay.memory) >= 1000:
					if self.eps > 0.01:
						self.eps -= 0.9 / self.epsilon_decay_steps
					self.dqn.train_network()

				score += reward
				state = next_state

				if terminal:
					self.dqn.update_target_network()
					scores.append(score)
					episodes.append(e)
					print('episode:', e, ' score:', score, ' epsilon', self.eps,
					      ' last 10 mean score', np.mean(scores[-min(10, len(scores)):]))

					if np.mean(scores[-min(10, len(scores)):]) > 9950:
						print('Already well trained')
						return
		pass

	def play(self):
		state = self.ENV.new_episode()
		self.ENV.render_worker(True)
		self.eps = 0.0

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
		checkpoint_dir = 'dqn_tensorflow'
		if not os.path.exists(checkpoint_dir):
			os.mkdir(checkpoint_dir)
		self.saver.save(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))

	def load(self):
		checkpoint_dir = 'dqn_tensorflow'
		self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))


if __name__ == "__main__":

	# parameter 저장하는 parser
	parser = argparse.ArgumentParser(description="CartPole")
	parser.add_argument('--env_name', default='CartPole-v1', type=str)
	parser.add_argument('--epsilon_decay_steps', default=2e4, type=int, help="how many steps for epsilon to be 0.1")
	parser.add_argument('--learning_rate', default=0.001, type=float)
	parser.add_argument('--batch_size', default=64, type=int)
	parser.add_argument('--discount_factor', default=0.99, type=float)
	parser.add_argument('--episodes', default=1000, type=float)
	sys.argv = ['-f']
	args = parser.parse_args()

	config = tf.ConfigProto()
	#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
	#config.log_device_placement = False
	#config.gpu_options.allow_growth = True

	# 학습 or 테스트
	with tf.Session(config=config) as sess:
		agent = Agent(args, sess)
		sess.run(tf.global_variables_initializer())  # tensorflow graph가 다 만들어지고 난 후에 해야됨
		#agent.train()
		#agent.save()
		agent.load()
		rewards = []
		for i in range(20):
			r = agent.play()
			rewards.append(r)
		mean = np.mean(rewards)
		print(rewards)
		print(mean)


