import os
import sys
import gym
import random
import numpy as np
import argparse
import tensorflow as tf
from collections import deque
import time
import copy


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
    def __init__(self, state_size, batch_size, length):
        self.length = length
        self.memory = deque(maxlen=self.length)
        self.state_size = state_size
        self.batch_size = batch_size
        pass

    def add(self, state, action, reward, next_state, terminal, gae):
        self.memory.append((state, action, reward, next_state, terminal, gae))
        pass

    def mini_batch(self):
        mini_batch = random.sample(self.memory, self.batch_size)  # memory에서 random하게 sample

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, terminals, gaes = [], [], [], []
        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            next_states[i] = mini_batch[i][3]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            terminals.append(mini_batch[i][4])
            gaes.append(mini_batch[i][5])
        return states, np.asarray(actions), rewards, next_states, terminals, gaes
        pass

    def clear(self):
        self.memory.clear()
        pass


class PPO(object):
    def __init__(self, state_size,  action_size, sess, learning_rate, discount_factor, replay, epsilon, a_bound):
        self.state_size = state_size
        self.action_size = action_size
        self.sess = sess
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.replay = replay
        self.eps = epsilon
        self.action_limit = a_bound

        self.state = tf.placeholder(tf.float32, [None, self.state_size])
        self.target = tf.placeholder(tf.float32, [None, 1])
        self.advantage = tf.placeholder(tf.float32, [None, 1])
        self.actions = tf.placeholder(tf.float32, [None, self.action_size])

        self.actor, self.sampled_action = self.build_actor('actor_eval', True)
        self.actor_target, _ = self.build_actor('actor_target', False)
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
        actor_hidden_size = 30
        with tf.variable_scope(scope):
            hidden1 = tf.layers.dense(self.state, actor_hidden_size, activation=tf.nn.relu, name='l1', trainable=trainable)
            m = tf.layers.dense(hidden1, self.action_size, activation=tf.nn.tanh, name='m', trainable=trainable)
            m = tf.multiply(m, self.action_limit, name='scaled_a')  # constrained mean value
            #m = tf.layers.dense(hidden1, self.action_size, name='m', trainable=trainable)  # [batch_size, action_size]
            std = 0.5 * tf.layers.dense(hidden1, self.action_size, activation=tf.nn.sigmoid, name='std', trainable=trainable)
            std = tf.add(std, tf.constant(0.5, shape=(self.replay.batch_size, self.action_size)))
            #std = tf.ones([self.replay.batch_size, self.action_size])
            output = tf.contrib.distributions.Normal(loc=m, scale=std)
            sampled_output = output.sample()
            return output, sampled_output  # [batch_size, action_size]
            pass

    def build_critic(self, scope, trainable):
        with tf.variable_scope(scope):
            critic_hidden_size = 30
            hidden1 = tf.layers.dense(self.state, critic_hidden_size, activation=tf.nn.relu, name='s1', trainable=trainable)
            hidden2 = tf.layers.dense(hidden1, critic_hidden_size, activation=tf.nn.relu, trainable=trainable)
            output = tf.layers.dense(hidden2, 1, trainable=trainable)
            return output
            pass

    def optimizer(self):
        policy = tf.clip_by_value(self.actor.prob(self.actions), 1e-10, 1.0)
        policy_old = tf.clip_by_value(self.actor_target.prob(self.actions), 1e-10, 1.0)

        ratio = policy / tf.add(policy_old, tf.constant(1e-10, shape=(self.replay.batch_size, 1)))
        #ratio = tf.exp(tf.log(policy) - tf.log(policy_old))

        min_a = ratio * self.advantage
        min_b = tf.clip_by_value(ratio, 1-self.eps, 1+self.eps) * self.advantage
        actor_loss = tf.reduce_mean(tf.math.minimum(min_a, min_b))
        critic_loss = tf.losses.mean_squared_error(labels=self.target, predictions=self.critic)
        entropy = -tf.reduce_sum(policy * tf.log(policy), axis=1)
        self.entropy = tf.reduce_mean(entropy, axis=0)
        loss = -actor_loss + critic_loss - 0.3 * self.entropy
        self.loss = -actor_loss + critic_loss
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return train_op, loss
        pass

    def train_network(self):
        states, actions, rewards, next_states, terminals, gaes = self.replay.mini_batch()

        next_target_v = self.sess.run(self.critic_target, feed_dict={self.state: next_states})

        target = []
        for i in range(self.replay.batch_size):
            if terminals[i]:
                target.append(rewards[i])
            else:
                target.append(rewards[i] + self.discount_factor * next_target_v[i])
        target = np.reshape(target, [self.replay.batch_size, 1])

        self.sess.run(self.train, feed_dict={self.state: states, self.advantage: gaes, self.actions: actions, self.target: target})
        #print(self.sess.run([0.3 * self.entropy, self.loss], feed_dict={self.state: states, self.advantage: gaes, self.actions: actions, self.target: target}))
        return self.sess.run(self.loss, feed_dict={self.state: states, self.advantage: gaes, self.actions: actions, self.target: target})
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
        self.a_bound = self.env.action_space.high[0]
        #self.env._max_episode_steps = 10000  # 최대 타임스텝 수
        self.epsilon = args.epsilon
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.discount_factor = args.discount_factor
        self.iteration = args.episodes

        self.actor = 8  # N
        self.timesteps = 500  # T
        self.gae_parameter = 0.95  # lambda
        self.epochs = 8  # K
        self.ENV = Environment(self.env, self.state_size, self.action_size)
        self.replay = ReplayMemory(self.state_size, self.batch_size, self.actor * self.timesteps)
        self.ppo = PPO(self.state_size, self.action_size, self.sess, self.learning_rate,
                       self.discount_factor, self.replay, self.epsilon, self.a_bound)
        self.saver = tf.train.Saver()
        pass

    def select_action(self, state):
        policy = self.sess.run(self.ppo.sampled_action, feed_dict={self.ppo.state: state})[0]
        policy_clip = np.clip(policy, -self.a_bound, self.a_bound)
        return policy_clip
        pass

    def make_delta(self, memory):
        states, rewards, next_states = [], [], []
        for i in range(len(memory)):
            states.append(memory[i][0])
            rewards.append(memory[i][2])
            next_states.append(memory[i][3])
        current_v = self.sess.run(self.ppo.critic, feed_dict={self.ppo.state: states})
        next_v = self.sess.run(self.ppo.critic, feed_dict={self.ppo.state: next_states})
        delta = [r_t + self.discount_factor * v_next - v for r_t, v_next, v in zip(rewards, next_v, current_v)]
        return delta
        pass

    def make_gae(self, memory):
        delta = self.make_delta(memory)
        gae = copy.deepcopy(delta)
        for t in reversed(range(len(gae) - 1)):
            gae[t] = gae[t] + self.gae_parameter * self.discount_factor * gae[t + 1]
            #memory[t].append(gae[t])
        #memory[len(gae)-1].append(gae[len(gae)-1])

        gae = np.array(gae).astype(np.float32)
        gae = (gae - gae.mean()) / gae.std()
        for t in range(len(gae)):
            memory[t].append(gae[t])
        pass

    def memory_to_replay(self, memory):
        self.make_gae(memory)
        for i in range(len(memory)):
            self.replay.add(memory[i][0], memory[i][1], memory[i][2], memory[i][3], memory[i][4], memory[i][5])
        pass

    def train(self):
        scores, losses, scores2 = [], [], []
        self.ppo.update_target_network()
        render = False
        for iteration in range(self.iteration):
            for _ in range(self.actor):
                memory, states, rewards, next_states = [], [], [], []
                score = 0
                terminal = False
                state = self.ENV.new_episode()
                for _ in range(self.timesteps):
                    self.ENV.render_worker(render)
                    #time.sleep(0.02)
                    state = np.reshape(state, [1, self.state_size])
                    action = self.select_action(state)
                    #action = self.sess.run(self.ppo.select_action(), feed_dict={self.ppo.state: state})[0]
                    next_state, reward, terminal = self.ENV.act(action)
                    state = state[0]
                    memory.append([state, action, reward / 5, next_state, terminal])
                    score += reward
                    state = next_state
                    if terminal:
                        break

                scores.append(score)
                self.memory_to_replay(memory)

            for _ in range(self.epochs):
                losses.append(self.ppo.train_network())

            self.ppo.update_target_network()
            self.replay.clear()
            scores2.append(np.mean(scores))
            print('episode:', iteration + 1, ' score:', int(np.mean(scores)), ' last 10 mean score', int(np.mean(scores2[-min(10, len(scores2)):])),
            ' loss', (np.mean(losses)))
            if int(np.mean(scores2[-min(10, len(scores2)):])) >= -400:
                iter1 = iteration
                render = True
                if int(np.mean(scores2[-min(10, len(scores2)):])) >= -250:
                    iter2 = iteration
                    print('Already Well Trained!!')
                    print(iter2 - iter1, 'iterations are needed for -400 to -250 score')
                    break
            else:
                render = False
            losses.clear()
            scores.clear()

        pass

    def play(self):
        state = self.ENV.new_episode()
        state = np.reshape(state, [1, self.state_size])

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
        checkpoint_dir = 'save_continuous'
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))

    def load(self):
        checkpoint_dir = 'save_continuous'
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))


if __name__ == "__main__":

    #print(sys.executable)
    # parameter 저장하는 parser
    parser = argparse.ArgumentParser(description="Pendulum")
    parser.add_argument('--env_name', default='Pendulum-v0', type=str)
    parser.add_argument('--learning_rate', default=0.003, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--discount_factor', default=0.95, type=float)
    parser.add_argument('--episodes', default=1000, type=float)
    parser.add_argument('--epsilon', default=0.2, type=float)
    sys.argv = ['-f']
    args = parser.parse_args()

    config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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


