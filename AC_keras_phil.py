from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model, load_model, model_from_json
from keras.optimizers import Adam
from keras.utils import plot_model
import numpy as np
import os

class ACKeras(object):
    def __init__(self,save_name ,alpha,beta,gamma=0.99,n_actions=4,
    layer1_size=1024,layer2_size=512,input_dims=8):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        self.save_name = save_name
        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        self.action_space = [i for i in range(self.n_actions)]
        
    def build_actor_critic_network(self):
        input = Input(shape=(self.input_dims,))
        delta = Input(shape=[1])
        self.delta_double = delta
        dense1 = Dense(self.fc1_dims,activation='relu')(input)
        dense2 = Dense(self.fc2_dims,activation='relu')(dense1)
        probs = Dense(self.n_actions,activation='softmax')(dense2)
        values = Dense(1,activation='linear')(dense2)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred,1e-8,1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*delta)

        actor = Model(input=[input,delta], output=[probs])
        actor.compile(optimizer=Adam(lr=self.alpha),loss=custom_loss)
        # actor.compile(optimizer=Adam(lr=self.alpha),loss='mean_squared_error')

        critic = Model(input=[input], output=[values])
        critic.compile(optimizer=Adam(lr=self.beta),loss='mean_squared_error')

        policy = Model(input=[input], output=[probs])

        return actor, critic, policy

    def Action(self, observation):
        state = observation[np.newaxis,:]
        probabilities = self.policy.predict(state)[0]
        action = np.random.choice(self.action_space,p=probabilities)
        
        return action

    def Train(self, action, state, state_, reward, done):
        state = state[np.newaxis,:]
        state_ = state_[np.newaxis,:]
        
        critic_value_ = self.critic.predict(state_)
        critic_value = self.critic.predict(state)

        target = reward + self.gamma*critic_value_*(1-int(done))
        delta = target - critic_value

        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1),action] = 1
        # print([state, delta])
        # print(actions)
        self.actor.fit([state, delta], actions, verbose=0)
        self.critic.fit(state, target, verbose=0)

    def SaveAgent(self):
        os.chdir('Agent_data')
        self.actor.save(self.save_name+'_actor')
        # actor_json = self.actor.to_json()
        # with open(self.save_name+"_actor.json", "w") as json_file:
        #     json_file.write(actor_json)
        # self.actor.save_weights(self.save_name+"_actor.h5")
        self.critic.save(self.save_name+'_critic')
        self.policy.save(self.save_name+'_policy')
        os.chdir('..')

    def LoadAgent(self):
        delta = self.delta_double
        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred,1e-8,1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*delta)
        os.chdir('Agent_data')
        self.critic = load_model(self.save_name+'_critic')
        self.policy = load_model(self.save_name+'_policy')
        
        # self.actor = load_model(self.save_name+'_actor')
        # json_file = open(self.save_name+'_actor.json', 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # self.actor = model_from_json(loaded_model_json)
        # self.actor.load_weights(self.save_name+'_actor.h5')
        # self.actor.compile(optimizer=Adam(lr=self.alpha),loss=custom_loss)

        # self.actor = model_from_json(open(self.save_name+'_actor.json').read())
        # self.actor.load_weights(self.save_name+'_actor.h5')
        # self.actor.compile(optimizer=Adam(lr=self.alpha),loss=custom_loss)

        self.actor = load_model(self.save_name+'_actor', custom_objects={'custom_loss': custom_loss})
        
        os.chdir('..')
        # plot_model(self.actor, to_file='AC_actor.png')
        # print(self.actor.get_config())

