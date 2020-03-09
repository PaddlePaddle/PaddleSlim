import numpy as np
import parl
from parl import layers
from paddle import fluid
from ..utils import RLCONTROLLER, action_mapping
from ..RLbase_controller import RLBaseController
from .ddpg_model import DefaultDDPGModel as default_ddpg_model
from parl.utils import ReplayMemory

__all__ = ['DDPG']


class DDPGAgent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(DDPGAgent, self).__init__(algorithm)

        # Attention: In the beginning, sync target model totally.
        self.alg.sync_target(decay=0)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.pred_act = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(
                name='act', shape=[self.act_dim], dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            _, self.critic_cost = self.alg.learn(obs, act, reward, next_obs,
                                                 terminal)

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act = self.fluid_executor.run(self.pred_program,
                                      feed={'obs': obs},
                                      fetch_list=[self.pred_act])[0]
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        critic_cost = self.fluid_executor.run(self.learn_program,
                                              feed=feed,
                                              fetch_list=[self.critic_cost])[0]
        self.alg.sync_target()
        return critic_cost


@RLCONTROLLER.register
class DDPG(RLBaseController):
    def __init__(self, **kwargs):
        self.obs_dim = kwargs.get('obs_dim')
        self.act_dim = kwargs.get('act_dim')
        self.model = kwargs.get(
            'model') if 'model' in kwargs else default_ddpg_model
        self.range_tables = kwargs.get('range_tables')
        self.actor_lr = kwargs.get(
            'actor_lr') if 'actor_lr' in kwargs else 1e-4
        self.critic_lr = kwargs.get(
            'critic_lr') if 'critic_lr' in kwargs else 1e-3
        self.gamma = kwargs.get('gamma') if 'gamma' in kwargs else 0.99
        self.tau = kwargs.get('tau') if 'tau' in kwargs else 0.001
        self.memory_size = kwargs.get(
            'memory_size') if 'memory_size' in kwargs else 10
        self.reward_scale = kwargs.get(
            'reward_scale') if 'reward_scale' in kwargs else 0.1
        self.batch_size = kwargs.get(
            'controller_batch_size') if 'controller_batch_size' in kwargs else 1

        model = self.model(self.act_dim)

        algorithm = parl.algorithms.DDPG(
            model,
            gamma=self.gamma,
            tau=self.tau,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr)
        self.agent = DDPGAgent(algorithm, self.obs_dim, self.act_dim)
        self.rpm = ReplayMemory(self.memory_size, self.obs_dim, self.act_dim)

    def next_tokens(self, states):
        batch_states = np.expand_dims(states, axis=0)
        actions = self.agent.predict(batch_states.astype('float32'))
        actions = action_mapping(actions, self.range_tables)
        actions = np.squeeze(actions)
        return actions

    def update(self, rewards, states, actions, states_next, terminal):
        self.rpm.append(states, actions, self.reward_scale * rewards,
                        states_next, terminal)
        if self.rpm.size() > self.memory_size:
            states, actions, rewards, states_next, terminal = rpm.sample_batch(
                self.batch_size)
        self.agent.learn(states, actions, rewards, states_next, terminal)
