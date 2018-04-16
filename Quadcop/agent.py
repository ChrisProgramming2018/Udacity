from collections import deque
import tensorflow as tf
import numpy as np




# Exploration parameters
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.0001            # exponential decay rate for exploration prob

        # Q-network learning rate


tf.reset_default_graph()

class Agent:
    def __init__(self, learning_rate=0.01, state_size=18, 
                 action_size=3, hidden_size=10, dropout=0.7,
                 name='QNetwork'):
        # state inputs to the Q-network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')
            
            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            #one_hot_actions = tf.one_hot(self.actions_, action_size)
            one_hot_actions = tf.one_hot(self.actions_, action_size)
            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            
            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            self.fc2 = tf.contrib.layers.dropout(self.fc1,dropout)
            self.fc3 = tf.contrib.layers.fully_connected(self.fc2, hidden_size)
            self.fc4 = tf.contrib.layers.dropout(self.fc3,dropout)
            self.fc5 = tf.contrib.layers.fully_connected(self.fc4, hidden_size)
            

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc5, action_size, 
                                                            activation_fn=None)
            
            ### Train with loss (targetQ - Q)^2
            # output has length 2, for two actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)
            
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
    
    def get_random_action(self):
        '''  '''
        # 0 stay 1 up 2 down
        # 2 down
        action = np.random.choice([0,1,2])
        return action




class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=False)
        return [self.buffer[ii] for ii in idx]