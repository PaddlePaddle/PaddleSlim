import parl
from ddpg_model import DefaultDDPGModel as default_ddpg_model


class DDPGAgent(parl.Agent):
    def __init__(self, obs_dim, act_dim):
        pass

    def build_program(self):
        pass

    def learn(self):
        pass

    def predict(self):
        pass


@RLCONTROLLER.register
class DDPG(RLBaseController):
    def __init__(self, *args, **kwargs):
        self.states_dim = kwargs.get('states_dim')
        self.token_dim = kwargs.get('token_dim')
        self.model = kwargs.get(
            'model') if 'model' in kwargs else default_ddpg_model
        self.obs_dim = kwargs.get('obs_dim')
        self.act_dim = kwargs.get('act_dim')

        algorithm = parl.algorithms.DDPG(self.model, and_so_on)
        self.agent = DDPGAgent(algorithm, self.obs_dim, self.act_dim)
        self.rpm = ReplayMemory(MEMORY_SIZE, self.obs_dim, self.act_dim)

    def next_tokens():
        actions = self.agent.predict(self.batch_obs.astype('float32'))
        ### map action to range table
        and_so_on
        return actions

    def reward(self, states, action, states_next, reward):
        self.rpm.append(obs, action, REWARD_SCALE * reward, next_obs, done)
        if rpm.size() > MIN_LEARN_SIZE:
            self.agent.learn(kwargs)
