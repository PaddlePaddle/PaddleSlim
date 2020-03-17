import math
import numpy as np
import paddle.fluid as fluid
from paddle.fluid import ParamAttr
from paddle.fluid.layers import RNNCell, LSTMCell, rnn
from paddle.fluid.contrib.layers import basic_lstm
from ..RLbase_controller import RLBaseController
from ..utils import RLCONTROLLER


uniform_initializer = lambda x: fluid.initializer.UniformInitializer(low=-x, high=x)


class lstm_cell(RNNCell):
    def __init__(self, num_layers, hidden_size):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm_cells = []

        param_attr = ParamAttr(initializer=uniform_initializer(
            1.0 / math.sqrt(hidden_size)))
        bias_attr = ParamAttr(initializer=uniform_initializer(
            1.0 / math.sqrt(hidden_size)))
        for i in range(num_layers):
            self.lstm_cells.append(
                LSTMCell(hidden_size, param_attr, bias_attr))

    def call(self, inputs, states):
        new_states = []
        for i in range(self.num_layers):
            out, new_state = self.lstm_cells[i](inputs, states[i])
            new_states.append(new_state)
        return out, new_states

    @property
    def state_shape(self):
        return [cell.state_shape for cell in self.lstm_cells]


@RLCONTROLLER.register
class LSTM(RLBaseController):
    def __init__(self, **kwargs):
        self.lstm_num_layers = kwargs.get('lstm_num_layers')
        self.hidden_size = kwargs.get('hidden_size')
        self.temperature = kwargs.get('temperature')
        self.range_tables = kwargs.get('range_tables')
        self.decay = kwargs.get('decay') if 'decay' in kwargs else 0.99
        self.weight_entropy = kwargs.get(
            'weight_entropy') if 'weight_entroy' in kwargs else None
        self.tanh_constant = kwargs.get(
            'tanh_constant') if 'tanh_constant' in kwargs else None
        #self.sample_entropy = fluid.layers.create_tensor(dtype='float32', name='sample_entropy', persistable=True) #0.0
        #self.sample_log_probs = fluid.layers.create_tensor(dtype='float32', name='sample_log_probs', persistable=True) #0.0
        self.baseline = 0.0

        self.max_range_table = max(self.range_tables) + 1

    def _lstm(self, inputs, hidden, cell, token_idx):
        cells = lstm_cell(self.lstm_num_layers, self.hidden_size)
        output, new_states = cells.call(inputs, states=([[cell, hidden]]))
        logits = fluid.layers.fc(new_states[0], self.range_tables[token_idx])

        logits = logits / self.temperature
        return logits, output, new_states

    def _network(self, inputs, hidden, cell):
        actions = []
        sample_entropies = []
        sample_log_probs = []

        for idx in range(len(self.range_tables)):
            logits, output, states = self._lstm(
                inputs, hidden, cell, token_idx=idx)
            hidden, cell = np.squeeze(states)
            probs = fluid.layers.softmax(logits, axis=1)
            action = fluid.layers.sampling_id(probs)
            log_prob = fluid.layers.cross_entropy(probs, action)
            sample_log_probs.append(log_prob)
            #self.sample_log_probs += fluid.layers.reduce_sum(log_prob)

            entropy = log_prob * fluid.layers.exp(-1 * log_prob)
            entropy.stop_gradient = True
            sample_entropies.append(entropy)
            #self.sample_entropy = fluid.layers.reduce_sum(entropy)

            action_emb = fluid.layers.cast(action, dtype=np.int64)
            emb_w = fluid.layers.create_parameter(
                name='emb_w',
                shape=(self.max_range_table, self.hidden_size),
                dtype='float32',
                default_initializer=fluid.initializer.Xavier())
            inputs = fluid.layers.gather(emb_w, action_emb)

            actions.append(action)

        sample_log_probs = fluid.layers.stack(sample_log_probs)
        self.sample_log_probs = fluid.layers.reduce_sum(sample_log_probs)

        return actions

    def _build_program(self,
                       main_program,
                       startup_program,
                       is_test=False,
                       batch_size=1):
        self.batch_size = batch_size
        with fluid.program_guard(main_program, startup_program):
            with fluid.unique_name.guard('Controller'):
                inputs = fluid.data(
                    name='inputs',
                    shape=[None, self.hidden_size],
                    dtype='float32')
                hidden = fluid.data(
                    name='hidden', shape=[None, self.hidden_size])
                cell = fluid.data(name='cell', shape=[None, self.hidden_size])
                tokens = self._network(inputs, hidden, cell)

                if is_test == False:
                    rewards = fluid.data(name='rewards', shape=[None])

                    avg_rewards = fluid.layers.reduce_mean(rewards)

                    if self.weight_entropy is not None:
                        avg_rewards += self.weight_entropy * self.sample_entropies

                    loss = avg_rewards * self.sample_log_probs
                    #self.baseline = self.baseline - (1.0 - self.decay) * (
                    #    self.baseline - avg_rewards)
                    #loss = self.sample_log_probs * (
                    #    avg_rewards - self.baseline)
                    optimizer = fluid.optimizer.Adam(learning_rate=0.1)
                    optimizer.minimize(loss)
                    return (inputs, hidden, rewards), tokens, loss

        return (inputs, hidden), tokens

    def _create_input(self, inputs, is_test=True, actual_rewards=None):
        feed_dict = dict()
        np_inputs = np.random.random(
            (self.batch_size, self.hidden_size)).astype('float32')
        np_init_hidden = np.zeros(
            (self.batch_size, self.hidden_size)).astype('float32')
        np_init_cell = np.zeros(
            (self.batch_size, self.hidden_size)).astype('float32')

        assert (len(inputs) == 2 or len(inputs) == 3), ""

        feed_dict["inputs"] = np_inputs
        feed_dict["hidden"] = np_init_hidden
        feed_dict["cell"] = np_init_cell

        if is_test == False:
            assert actual_rewards != None, "if you want to update controller, you must inputs a reward"
            if isinstance(actual_rewards, np.float):
                actual_rewards = np.expand_dims(actual_rewards, axis=0)

            feed_dict['rewards'] = actual_rewards

        return feed_dict

    def next_tokens(self, num_archs=1, params_dict=None):
        """ sample next tokens according current parameter and inputs"""
        main_program = fluid.Program()
        startup_program = fluid.Program()
        inputs, tokens = self._build_program(
            main_program, startup_program, is_test=True, batch_size=num_archs)

        place = fluid.CUDAPlace(0)  #if self.args.use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)

        for var in main_program.global_block().all_parameters():
            fluid.global_scope().find_var(var.name).get_tensor().set(
                params_dict[var.name], place)

        feed_dict = self._create_input(inputs)

        actions = exe.run(main_program, feed=feed_dict, fetch_list=tokens)

        batch_tokens = []
        for idx in range(self.batch_size):
            each_token = {}
            for i, action in enumerate(actions):
                token = action[idx]  #.asscalar()
                if idx in each_token:
                    each_token[idx].append(int(token))
                else:
                    each_token[idx] = [int(token)]
            batch_tokens.append(each_token[idx])

        ### return batch config type [[]], but search space receive []
        return np.squeeze(batch_tokens)

    def update(self, rewards, params_dict):
        """train controller according reward"""
        main_program = fluid.Program()
        startup_program = fluid.Program()
        inputs, tokens, loss = self._build_program(main_program,
                                                   startup_program)

        place = fluid.CUDAPlace(0)  #if self.args.use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)

        self.set_params(main_program, params_dict, place)

        feed_dict = self._create_input(
            inputs, is_test=False, actual_rewards=rewards)

        build_strategy = fluid.BuildStrategy()
        compiled_program = fluid.CompiledProgram(
            main_program).with_data_parallel(
                loss.name, build_strategy=build_strategy)

        fetch_list = tokens
        outs = exe.run(compiled_program, feed=feed_dict, fetch_list=fetch_list)
        tokens = []
        for o in outs:
            tokens.append(o[0])
        print('tokens: ', tokens)
        params_dict = self.get_params(main_program)
        return params_dict
