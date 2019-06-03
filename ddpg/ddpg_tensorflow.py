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
	def __init__(self, env, state_size):
		self.env = env
		self.state_size = state_size
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
	def __init__(self, state_size,  sess, learning_rate_actor, learning_rate_critic,
	             replay, discount_factor):
		self.state_size = state_size
		self.sess = sess
		self.lr_actor = learning_rate_actor
		self.lr_critic = learning_rate_critic
		self.replay = replay
		self.discount_factor = discount_factor

		self.state = tf.placeholder(tf.float32, [None, self.state_size])
		self.target = tf.placeholder(tf.float32, [None])

		self.actor = self.build_actor('actor')
		self.actor_target = self.build_actor('actor_target')
		self.critic = self.build_critic('critic')
		self.critic_target = self.build_critic_target('critic_target')
		self.train_actor = self.actor_optimizer()
		self.train_critic = self.critic_optimizer()
		pass

	def build_actor(self, name):
		with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
			h1 = tf.layers.dense(self.state, 24, activation=tf.nn.relu)
			output = tf.layers.dense(h1, 1)
			return output
			pass

	def build_critic(self, name):
		with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
			h1 = tf.layers.dense(self.state, 24) + tf.layers.dense(self.actor, 24)
			h1_relu = tf.nn.relu(h1)
			output = tf.layers.dense(h1_relu, 1)
			return output
			pass

	def build_critic_target(self, name):
		with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
			h1 = tf.layers.dense(self.state, 24) + tf.layers.dense(self.actor_target, 24)
			h1_relu = tf.nn.relu(h1)
			output = tf.layers.dense(h1_relu, 1)
			return output
			pass

	def actor_optimizer(self):
		actor_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')

		loss = tf.reduce_mean(self.critic)
		train_op = tf.train.AdamOptimizer(-self.lr_actor).minimize(loss, var_list=actor_parameters)

		return train_op
		pass

	def critic_optimizer(self):
		critic_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')

		loss = tf.reduce_mean(tf.square(self.target - self.critic))    # mse loss
		train_op = tf.train.AdamOptimizer(self.lr_critic).minimize(loss, var_list=critic_parameters)
		return train_op
		pass

	def train_network(self):
		states, actions, rewards, next_states, terminals = self.replay.mini_batch()

		next_target_q = self.sess.run(self.critic_target, feed_dict={self.state: next_states})

		target = []
		for i in range(self.replay.batch_size):
			if terminals[i]:
				target.append(rewards[i][0])
			else:
				target.append(rewards[i][0] + self.discount_factor * next_target_q[i][0])

		self.sess.run(self.train_actor, feed_dict={self.state: states})
		self.sess.run(self.train_critic, feed_dict={self.state: states, self.target: target})
		pass

	def update_target_network(self):
		copy_op = []
		actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
		actor_target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_target')
		for actor_var, actor_target_var in zip(actor_vars, actor_target_vars):
			copy_op.append(actor_target_var.assign(actor_var.value()))
		self.sess.run(copy_op)

		copy_op = []
		critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
		critic_target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_target')
		for critic_var, critic_target_var in zip(critic_vars, critic_target_vars):
			copy_op.append(critic_target_var.assign(critic_var.value()))
		self.sess.run(copy_op)


class Agent(object):
	def __init__(self, args, sess):
		# CartPole 환경
		self.env = gym.make(args.env_name)
#		self.eps = 1.0  # epsilon
		self.sess = sess
		self.state_size = self.env.observation_space.shape[0]
#		self.action_size = self.env.action_space.n
#		self.env._max_episode_steps = 10000  # 최대 타임스텝 수 10000
#		self.epsilon_decay_steps = args.epsilon_decay_steps
		self.learning_rate = args.learning_rate
		self.batch_size = args.batch_size
		self.discount_factor = args.discount_factor
		self.episodes = args.episodes
		self.ENV = Environment(self.env, self.state_size)
		self.replay = ReplayMemory(self.env, self.state_size, self.batch_size)
		self.ddpg = DDPG(self.state_size, self.sess, self.learning_rate[0], self.learning_rate[1],
		               self.replay, self.discount_factor)
		self.saver = tf.train.Saver()
		self.action_variance = 3
		pass

	def select_action(self, state):
		#action = self.sess.run(self.ddpg.actor, feed_dict={self.ddpg.state: state})
		return np.clip(
			np.random.normal(self.sess.run(self.ddpg.actor, {self.ddpg.state: state}), self.action_variance), -2,
			2)
		#return action
		pass

	def train(self):
		scores, episodes = [], []
		for e in range(self.episodes):
			terminal = False
			score = 0
			step = 0
			state = self.ENV.new_episode()
			state = np.reshape(state, [1, self.state_size])

			while not terminal:
				action = self.select_action(state)
				next_state, reward, terminal = self.ENV.act(action)
				next_state = np.reshape(next_state, [1, self.state_size])
				self.replay.add(state, action, reward, next_state, terminal)

				if len(self.replay.memory) >= 1000:
					self.ddpg.train_network()
					self.action_variance *= .9995

				score += reward[0]
				state = next_state
				step += 1

				if step % 1000 == 0:
					self.ddpg.update_target_network()

				if terminal:
					self.ddpg.update_target_network()
					scores.append(score)
					episodes.append(e)
					print('episode:', e+1, ' score:', score, ' last 10 mean score', np.mean(scores[-min(10, len(scores)):]))

					if np.mean(scores[-min(10, len(scores)):]) > 5000:
						print('Already well trained')
						return
		pass

	def play(self):
		state = self.ENV.new_episode()
		self.ENV.render_worker(True)

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
		checkpoint_dir = 'save'
		if not os.path.exists(checkpoint_dir):
			os.mkdir(checkpoint_dir)
		self.saver.save(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))

	def load(self):
		checkpoint_dir = 'save'
		self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))


if __name__ == "__main__":

	# parameter 저장하는 parser
	parser = argparse.ArgumentParser(description="Pendulum")
	parser.add_argument('--env_name', default='Pendulum-v0', type=str)
#	parser.add_argument('--epsilon_decay_steps', default=7e4, type=int, help="how many steps for epsilon to be 0.1")
	parser.add_argument('--learning_rate', default=[0.002, 0.001], type=list)
	parser.add_argument('--batch_size', default=64, type=int)
	parser.add_argument('--discount_factor', default=0.99, type=float)
	parser.add_argument('--episodes', default=1000, type=float)
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
			rewards.append(r)
		mean = np.mean(rewards)
		print(rewards)
		print(mean)


