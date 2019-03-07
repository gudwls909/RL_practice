import cv2
import os
import sys
import gym
import random
import numpy as np
import argparse
import tensorflow as tf
from collections import deque


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


class A2C(object):
	def __init__(self, state_size, action_size, sess, learning_rate_actor,
	             learning_rate_critic, discount_factor):
		self.state_size = state_size
		self.action_size = action_size
		self.sess = sess
		self.lr_actor = learning_rate_actor
		self.lr_critic = learning_rate_critic
#		self.replay = replay
		self.discount_factor = discount_factor

		self.state = tf.placeholder(tf.float32, [1, self.state_size])
		self.action = tf.placeholder(tf.float64, [None])
		self.target = tf.placeholder(tf.float32, [1])
		self.advantage = tf.placeholder(tf.float32, [1])

		self.actor = self.build_actor('actor')
		self.critic = self.build_critic('critic')
		self.train_actor = self.actor_optimizer()
		self.train_critic = self.critic_optimizer()
		pass

	def build_actor(self, name):
		with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
			h1 = tf.layers.dense(self.state, 25, activation=tf.nn.relu,
			                     kernel_initializer=tf.initializers.truncated_normal)
			output = tf.layers.dense(h1, self.action_size,
			                     kernel_initializer=tf.initializers.truncated_normal)
			return output
			pass

	def build_critic(self, name):
		with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
			h1 = tf.layers.dense(self.state, 25, activation=tf.nn.relu,
			                     kernel_initializer=tf.initializers.truncated_normal)
			h2 = tf.layers.dense(h1, 25, activation=tf.nn.relu,
			                     kernel_initializer=tf.initializers.truncated_normal)
			output = tf.layers.dense(h2, 1,
			                     kernel_initializer=tf.initializers.truncated_normal)
			return output
			pass

	def actor_optimizer(self):
		opt = tf.train.AdamOptimizer(self.lr_actor)
		actor_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
		grads_action = opt.compute_gradients(tf.log(self.actor), var_list=actor_parameters)  # actor parameter의 gradient만 원해서
		'''
		grads_score = []
		print(self.advantage, grads_action[0][0])
		print(grads_action[0])
		for i in range(len(grads_action)):
			grads_score.append((grads_action[i][0] * self.advantage, grads_action[i][1] * self.advantage))
		#grads_score = grads_action.dot(self.advantage)
		print(grads_score[0])
		train_op = opt.apply_gradients(grads_score)
		print('a')
		''' #뭔가안됨
		return train_op
		pass

	def critic_optimizer(self):
		loss = tf.reduce_mean(tf.square(self.advantage))    #  mse loss
		train_op = tf.train.AdamOptimizer(self.lr_critic).minimize(loss)
		return loss, train_op
		pass

	def train_network(self, state, action, reward, next_state, terminal):
		#  batch size = 1
		current_Q = self.sess.run(self.critic, feed_dict={self.state: state})
		next_Q = self.sess.run(self.critic, feed_dict={self.state: next_state})
		if terminal:
			advantage = reward - current_Q
			target = append(reward)
		else:
			advantage = reward + self.discount_factor * next_Q - current_Q
			target = reward + self.discount_factor * next_Q

		self.sess.run(self.train_actor, feed_dict={self.state: state, self.action: action, self.advantage: advantage})
		self.sess.run(self.train_critic, feed_dict={self.state: state, self.action: action, self.target: target})
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
#		self.batch_size = args.batch_size
		self.discount_factor = args.discount_factor
		self.episodes = args.episodes
		self.ENV = Environment(self.env, self.state_size, self.action_size)
		self.a2c = A2C(self.state_size, self.action_size, self.sess, self.learning_rate[0], self.learning_rate[1],
		               self.discount_factor)
		self.saver = tf.train.Saver()
		pass

	def select_action(self, state):
		self.sess.run(self.a2c.actor, feed_dict={self.a2c.state: state})
		return np.random.choice(self.action_size, 1, p=policy)[0]
		pass

	def train(self):
		scores, episodes = [], []
		for e in range(self.episodes):
			terminal = False
			score = 0
			state = self.ENV.new_episode()
			state = np.reshape(state, [1, self.state_size])

			while not terminal:
				action = self.select_action(state)
				next_state, reward, terminal = self.ENV.act(action)
				next_state = np.reshape(next_state, [1, self.state_size])
#				self.replay.add(state, action, reward, next_state, terminal)

				if len(self.replay.memory) >= 1000:
					if self.eps > 0.1:
						self.eps -= 0.9 / self.epsilon_decay_steps
					self.a2c.train_network(state, action, reward, next_state, terminal)

				score += reward
				state = next_state

				if terminal:
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
		checkpoint_dir = 'a2c_tensorflow'
		if not os.path.exists(checkpoint_dir):
			os.mkdir(checkpoint_dir)
		self.saver.save(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))

	def load(self):
		checkpoint_dir = 'a2c_tensorflow'
		self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))


if __name__ == "__main__":

	# parameter 저장하는 parser
	parser = argparse.ArgumentParser(description="CartPole")
	parser.add_argument('--env_name', default='CartPole-v1', type=str)
	parser.add_argument('--epsilon_decay_steps', default=1e4, type=int, help="how many steps for epsilon to be 0.1")
	parser.add_argument('--learning_rate', default=[0.001, 0.001], type=list)
	#parser.add_argument('--batch_size', default=64, type=int)
	parser.add_argument('--discount_factor', default=0.99, type=float)
	parser.add_argument('--episodes', default=100, type=float)
	sys.argv = ['-f']
	args = parser.parse_args()

	config = tf.ConfigProto()
	#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
	#config.log_device_placement = False
	#config.gpu_options.allow_growth = True

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
			rewards.append(r)
		mean = np.mean(rewards)
		print(rewards)
		print(mean)


