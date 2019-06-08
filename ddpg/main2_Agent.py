
import tensorflow as tf
import numpy as np
from replayBuffer import ReplayBuffer

class Agent_DDPG(object):
    def __init__(self, action_size, state_size, action_limit,):
        self.memory_size = 10000
        self.replayBuffer = ReplayBuffer(self.memory_size)
        self.sess = tf.Session()

        self.discount_factor = 0.9
        self.action_variance = 3
        self.critic_learning_rate = 0.001
        self.actor_learning_rate = 0.002
        self.batch_size = 32
        
        self.action_size, self.state_size, self.action_limit = action_size, state_size, action_limit,
        self.input_state = tf.placeholder(tf.float32, [None, state_size], 's')
        self.input_state_ = tf.placeholder(tf.float32, [None, state_size], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self.build_actor_network(self.input_state, scope='eval', trainable=True)
            a_ = self.build_actor_network(self.input_state_, scope='tar', trainable=False)
        with tf.variable_scope('Critic'):
            q_eval = self.build_critic_network(self.input_state, self.a, scope='eval', trainable=True)
            q_target = self.build_critic_network(self.input_state_, a_, scope='target', trainable=False)

        self.actor_evaluation_params  = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.actor_target_params      = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/tar')
        self.critic_evaluation_params = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.critic_target_params     = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/tar')

        self.replace = [tf.assign(t, (1 - 0.01 ) * t + 0.01  * e)
                             for t, e in zip(self.actor_target_params + self.critic_target_params,
                                             self.actor_evaluation_params + self.critic_evaluation_params)]

        '''
               dJ/dtheta = E[ dQ/dtheta ] 

        '''
        # Actor Loss 는 Q로부터 내려오는 값을 maximize 하면 된다(논문 참조)
        self.a_loss = tf.reduce_mean(q_eval)  # maximize the q
        # Maximize Q 를 해야하므로 learning rate에 '-' 를 붙인다.
        self.atrain = tf.train.AdamOptimizer(-self.actor_learning_rate).minimize(tf.reduce_mean(q_eval),
                                                                                 var_list=self.actor_evaluation_params)

        # self.c_train 을 호출할때 self.a 에 배치의 action을 넣게 된다.
        # Placeholder가 아닌 self.a 에 직접 값을 대입하는 것!
        # s a r s_ 를 이용해서 critic을 업데이트 하는데, 정석으로 구한 y가 트루 라벨, 뉴럴넷에 값을 넣고 나오는 것이 우리의 prediction이다.
        # True Label,  y = r(s,u_t(s)) + gamma*Q(s_, u_t(s_))
        q_true = self.R + self.discount_factor * q_target

        # Prediction, Q = q_eval
        # 우리가 mseLoss를 구하려면 q_eval을 구해야 하므로 self.input_state에 피딩을 해 주어야 함.
        # 또한 q_true 를 구하기 위해 self.R 과 q_target에 들어갈 self.input_state_ 도 피딩 해주어야 함.
        self.mseloss = tf.losses.mean_squared_error(labels=q_true, predictions=q_eval)
        # 이 부분은 오직 Critic net을 업데이트하기위한 Loss이다. 때문에 var_list를 Critic evaluation network로 지정해주어야한다.
        self.ctrain = tf.train.AdamOptimizer(self.critic_learning_rate).minimize(self.mseloss, var_list=self.critic_evaluation_params)


        # 네트워크를 만들고 항상 초기화를 해준다.
        self.sess.run(tf.global_variables_initializer())

        self.actor_loss_history = []
        self.critic_loss_history = []

    def store_transition(self, s, a, r, s_):
        self.replayBuffer.add(s,a,r,s_)

    def choose_action(self, s):
        return np.clip(np.random.normal(self.sess.run(self.a, {self.input_state: s[np.newaxis, :]})[0] , self.action_variance), -2, 2)

    def learn(self):
        if self.replayBuffer.count() > self.batch_size:
            self.action_variance *= .9995
            self.sess.run(self.replace)

            batch = self.replayBuffer.get_batch(self.batch_size)
            batch_s = np.asarray([x[0] for x in batch])
            batch_a = np.asarray([x[1] for x in batch])
            batch_r = np.asarray([[x[2]] for x in batch])
            batch_s_ = np.asarray([x[3] for x in batch])

            actor_loss, _ = self.sess.run([self.a_loss, self.atrain], {self.input_state: batch_s})
            critic_loss, _ = self.sess.run([self.mseloss, self.ctrain], {self.input_state: batch_s, self.a: batch_a, self.R: batch_r, self.input_state_: batch_s_})

            self.actor_loss_history.append(actor_loss)
            self.critic_loss_history.append(critic_loss)

    def build_actor_network(self, s, scope, trainable):
        actor_hidden_size = 30
        with tf.variable_scope(scope):
            hidden1 = tf.layers.dense(s, actor_hidden_size, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(hidden1, self.action_size, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.action_limit, name='scaled_a')

    def build_critic_network(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            critic_hidden_size = 30
            hidden1 = tf.layers.dense(s, critic_hidden_size,  name='s1', trainable=trainable) \
            + tf.layers.dense(a, critic_hidden_size, name='a1', trainable=trainable) \
            + tf.get_variable('b1', [1, critic_hidden_size], trainable=trainable)
            hidden1 = tf.nn.relu(hidden1)
            return tf.layers.dense(hidden1, 1, trainable=trainable)

    def plot_loss(self):
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.font_manager._rebuild()
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams.update({'font.size': 25})
        matplotlib.rc('text', usetex=True)
        plt.title('$\mathit{history}$', fontsize=25)
        ms = 0.1
        me = 1
        line_width = 0.1
        plt.ylabel('Loss')
        plt.xlabel('Training steps')

        actor_loss_mean = sum(self.actor_loss_history)/len(self.actor_loss_history)
        self.actor_loss_history /= actor_loss_mean
        critic_loss_mean = sum(self.critic_loss_history)/len(self.critic_loss_history)
        self.critic_loss_history /= critic_loss_mean

        plt.plot(np.arange(len(self.actor_loss_history)), self.actor_loss_history, '-p', color='b', markevery=me, label=r'actor loss', lw=line_width,
                 markersize=ms)
        plt.plot(np.arange(len(self.critic_loss_history)), self.critic_loss_history, '--^', color='r', markevery=me, label=r'critic loss', lw=line_width, markersize=ms)

        plt.grid()
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylim(0, 10)
        plt.show()

    def plot_reward(self, reward_history):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(reward_history)), reward_history)
        plt.ylabel('Reward')
        plt.xlabel('Episodes')
        plt.grid()
        plt.show()

