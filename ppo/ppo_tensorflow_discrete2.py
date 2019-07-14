import os
import sys
import gym
import random
import numpy as np
import argparse
import tensorflow as tf
from collections import deque
import copy


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


class PPO(object):
    def __init__(self, state_size,  action_size, sess, learning_rate, discount_factor, epsilon):
        self.state_size = state_size
        self.action_size = action_size
        self.sess = sess
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.eps = epsilon
        self.eps_len = 0

        self.state = tf.placeholder(tf.float32, [None, self.state_size])
        self.advantage = tf.placeholder(tf.float32, [None, 1])
        self.actions = tf.placeholder(tf.int32, [None, 1])
        self.target = tf.placeholder(tf.float32, [None, 1])

        self.actor = self.build_actor('actor_eval', True)
        self.actor_target = self.build_actor('actor_target', False)
        self.critic = self.build_critic('critic_eval', True,)
        self.critic_target = self.build_critic('critic_target', False)

        self.actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_eval')
        self.actor_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_target')
        self.critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_eval')
        self.critic_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_target')

        self.replace = [tf.assign(t, e)
                   for t, e in zip(self.actor_target_vars + self.critic_target_vars, self.actor_vars + self.critic_vars)]

        self.train, self.loss = self.optimizer()
        pass

    def build_actor(self, scope, trainable):
        actor_hidden_size = 20
        with tf.variable_scope(scope):
            hidden1 = tf.layers.dense(self.state, actor_hidden_size, activation=tf.nn.relu, name='l1', trainable=trainable)
            hidden2 = tf.layers.dense(hidden1, actor_hidden_size, activation=tf.nn.relu, name='l2', trainable=trainable)
            output = tf.layers.dense(hidden2, self.action_size, activation=tf.nn.softmax, name='m', trainable=trainable)

            #act_stochastic = tf.random.multinomial(tf.log(act_probs), num_samples=1)
            #act_stochastic = tf.reshape(act_stochastic, shape=[-1])

            return output
            pass

    def build_critic(self, scope, trainable):
        with tf.variable_scope(scope):
            critic_hidden_size = 20
            hidden1 = tf.layers.dense(self.state, critic_hidden_size, activation=tf.nn.relu, name='s1', trainable=trainable)
            hidden2 = tf.layers.dense(hidden1, critic_hidden_size, activation=tf.nn.relu, trainable=trainable)
            output = tf.layers.dense(hidden2, 1, trainable=trainable)
            return output
            pass

    def optimizer(self):
        policy = self.actor * tf.one_hot(self.actions, self.action_size)
        policy = tf.reduce_sum(policy, axis=2)
        policy_old = self.actor_target * tf.one_hot(self.actions, self.action_size)
        policy_old = tf.reduce_sum(policy_old, axis=2)
        ratio = policy / tf.add(policy_old, tf.constant(1e-10, shape=(self.eps_len, 1)))
        #ratio = tf.exp(tf.log(policy) - tf.log(policy_old))

        min_a = ratio * self.advantage
        min_b = tf.clip_by_value(ratio, 1-self.eps, 1+self.eps) * self.advantage
        actor_loss = tf.reduce_mean(tf.math.minimum(min_a, min_b))
        critic_loss = tf.losses.mean_squared_error(labels=self.target, predictions=self.critic)
        entropy = -tf.reduce_sum(policy * tf.log(tf.clip_by_value(policy, 1e-10, 1.0)), axis=1)
        entropy = tf.reduce_mean(entropy, axis=0)
        loss = -actor_loss + critic_loss - 0.01 * entropy
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return train_op, loss
        pass

    def train_network(self, states, actions, rewards, next_states):
        current_v = self.sess.run(self.critic, feed_dict={self.state: states})
        next_target_v = self.sess.run(self.critic_target, feed_dict={self.state: next_states})

        gaes = self.get_gaes(rewards, current_v, next_target_v)

        target = []
        for i in range(len(rewards)):
            if i == len(rewards)-1:
                target.append(rewards[i])
            else:
                target.append(rewards[i] + self.discount_factor * next_target_v[i])
        target = np.reshape(target, [len(rewards), 1])
        gaes = np.reshape(gaes, [len(rewards), 1])

        self.eps_len = len(rewards)

        self.sess.run(self.train, feed_dict={self.state: states, self.actions: actions, self.target: target, self.advantage: gaes})
        return self.sess.run(self.loss, feed_dict={self.state: states, self.actions: actions, self.target: target, self.advantage: gaes})
        pass

    def update_target_network(self):
        self.sess.run(self.replace)
        pass

    def get_gaes(self, rewards, v_preds, v_preds_next):
        deltas = [r_t + self.discount_factor * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.discount_factor * gaes[t + 1]
        return gaes


class Agent(object):
    def __init__(self, args, sess):
        # CartPole 환경
        self.env = gym.make(args.env_name)
        self.sess = sess
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.env._max_episode_steps = 10000  # 최대 타임스텝 수 10000
        self.learning_rate = args.learning_rate
        self.epsilon = args.epsilon
        self.discount_factor = args.discount_factor
        self.episodes = args.episodes
        self.ENV = Environment(self.env, self.state_size, self.action_size)
        self.ppo = PPO(self.state_size, self.action_size, self.sess, self.learning_rate, self.discount_factor, self.epsilon)
        self.saver = tf.train.Saver()
        pass

    def select_action(self, state):
        policy = self.sess.run(self.ppo.actor, feed_dict={self.ppo.state: state}).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]
        pass

    def train(self):
        scores, episodes, losses = [], [], []
        states, actions, rewards, next_states = [], [], [], []
        for e in range(self.episodes):
            terminal = False
            score = 0
            state = self.ENV.new_episode()[0]

            while not terminal:
                states.append(state)
                state = np.reshape(state, [1, self.state_size])
                action = self.select_action(state)
                next_state, reward, terminal = self.ENV.act(action)

                actions.append([action])
                rewards.append(reward)
                next_states.append(next_state)

                score += reward
                state = next_state

            loss = self.ppo.train_network(states, actions, rewards, next_states)
            losses.append(loss)

            scores.append(score)
            episodes.append(e)
            print('episode:', e, ' score:', int(score), ' last 10 mean score', int(np.mean(scores[-min(10, len(scores)):])),
                  ' loss', (np.mean(losses[-min(10, len(losses)):])))
        pass

    def play(self):
        state = self.ENV.new_episode()

        terminal = False
        score = 0
        while not terminal:
            self.ENV.render_worker(True)
            action = self.select_action(state)
            next_state, reward, terminal = self.ENV.act(action)
            next_state = np.reshape(next_state, [1, self.state_size])
            score += reward
            state = next_state

            if terminal:
                return score
        pass

    def save(self):
        checkpoint_dir = 'asdf'
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))

    def load(self):
        checkpoint_dir = 'asdf'
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))


if __name__ == "__main__":

    # parameter 저장하는 parser
    parser = argparse.ArgumentParser(description="CartPole")
    parser.add_argument('--env_name', default='CartPole-v1', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=list)
    parser.add_argument('--discount_factor', default=0.95, type=float)
    parser.add_argument('--episodes', default=1000, type=float)
    parser.add_argument('--epsilon', default=0.2, type=float)

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


