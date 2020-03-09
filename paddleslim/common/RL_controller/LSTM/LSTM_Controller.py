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
        self.total_token_num = sum(self.range_tables)
        self.with_entropy = kwargs.get(
            'with_entropy') if 'with_entroy' in kwargs else False

    def _lstm(self, inputs, hidden, cell, is_embed, token_idx):
        #if not is_embed:
        #    #inputs = fluid.embedding(self.total_token_num, self.hidden_size)
        #    inputs_1 = fluid.embedding(inputs, self.hidden_size)
        #else:
        inputs_1 = inputs
        #print("---------------------", token_idx)
        #print("======================", inputs_1)
        cells = lstm_cell(self.lstm_num_layers, self.hidden_size)
        #output, new_hidden = inputs, inputs 
        output, new_hidden = rnn(cell=cells,
                                 inputs=inputs_1,
                                 initial_states=([[cell, hidden]]))
        logits = fluid.layers.fc(new_hidden[0], self.range_tables[token_idx])

        logits = logits / self.temperature
        return logits, output, new_hidden

    def _network(self, inputs, hidden, cell):
        actions = []
        entropies = []
        log_probs = []
        all_h = []
        for idx in range(len(self.range_tables)):
            logits, output, hidden = self._lstm(
                inputs, hidden, cell, is_embed=(idx == 0), token_idx=idx)
            all_h.append(hidden[-1])
            probs = fluid.layers.softmax(logits, axis=1)
            log_prob = fluid.layers.log(probs)
            entropy = -1 * fluid.layers.sum(
                (log_prob * probs), axis=1) if self.with_entropy else None

            ### don't have this op
            #action = np.random.multinomial(1, np_probs)
            action = fluid.layers.sampling_id(probs)
            index = fluid.layers.stack(probs, axis=0)
            #seleted_log_prob = fluid.layers.gather_nd(log_prob, index)
            selected_log_prob = log_prob

            actions.append(action)
            #actions.append(action[:, 0])
            entropies.append(entropy)
            log_probs.append(selected_log_prob)
            #log_probs = None

            #print("-----------------------", all_h, action)
            inputs = fluid.layers.gather(
                all_h[:-1], action)  #action + sum(self.range_tables[:idx])
            #inputs = action[:, 0] + sum(self.range_tables[:idx])

        tokens = []
        for idx in range(self.batch_size):
            each_token = {}  ###batch_size: [token]
            for i, action in enumerate(actions):
                token = action[idx].asscalar()
                if idx in each_token:
                    each_token[idx].append(int(token))
                else:
                    each_token[idx] = [int(token)]
            tokens.append(each_token)

        entropies = fluid.layers.stack(
            *entropies, axis=1) if with_entropy else entropies
        return tokens, fluid.layers.stack(*log_probs, axis=1), entropies

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
                    shape=[None, self.batch_size, self.hidden_size],
                    dtype='float32')
                hidden = fluid.data(
                    name='hidden', shape=[None, self.hidden_size])
                cell = fluid.data(name='cell', shape=[None, self.hidden_size])
                tokens, log_probs, entropies = self._network(inputs, hidden,
                                                             cell)

                if is_test == False:
                    rewards = fluid.data(name='rewards', shape=[None, 1])

                    for idx in range(self.batch_size):
                        self.baseline = reward if not self.baseline else self.baseline
                        avg_reward = reward - self.baseline
                        self.baseline = self.decay * self.baseline + (
                            1.0 - self.decay) * rewards
                        log_prob = fluid.layers.gather(log_probs, idx)
                        log_prob = fluid.layers.reduce_sum(log_prob)
                        loss = -1 * log_prob * avg_reward
                        optimizer = fluid.optimizer.Adam(learning_rate=0.1)
                        optimizer.minimize(loss)
                return (inputs, hidden, rewards), tokens, loss

        return (inputs, hidden), tokens, loss

    def _create_input(self, inputs, is_test=True, actual_rewards=None):
        feed_dict = dict()
        np_inputs = np.zeros(
            (self.batch_size, self.hidden_size))  ### key == batch_size[i]
        np_init_hidden = np.zeros((self.batch_size, self.hidden_size))

        assert (len(inputs) == 2 or len(inputs) == 3), ""

        feed_dict["inputs"] = np_inputs
        feed_dict["hidden"] = np_init_hidden

        if is_test == False:
            assert actual_reward != None, "if you want to update controller, you must inputs a reward"
            feed_dict[rewards] = actual_rewards

        return feed_dict

    def next_tokens(self, num_archs=1):
        """ sample next tokens according current parameter and inputs"""
        main_program = fluid.Program()
        startup_program = fluid.Program()
        inputs, tokens, loss = self._build_program(
            main_program, startup_program, is_test=True, batch_size=num_archs)

        place = fluid.CUDAPlace(0) if self.args.use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)

        build_strategy = fluid.BuildStrategy()
        compiled_program = fluid.CompiledProgram(
            main_program).with_data_parallel(
                loss.name, build_strategy=build_strategy)
        feed_dict = self._create_input(inputs)

        token = exe.run(compiled_program, feed=feed_dict, fetch_list=[tokens])
        return token

    def update(self, rewards):
        """train controller according reward"""
        main_program = fluid.Program()
        startup_program = fluid.Program()
        inputs, tokens, loss = self._build_program(main_program,
                                                   startup_program)

        place = fluid.CUDAPlace(0) if self.args.use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)

        feed_dict = self._create_input(
            inputs, is_test=False, actual_rewards=reward)

        build_strategy = fluid.BuildStrategy()
        compiled_program = fluid.CompiledProgram(
            main_program).with_data_parallel(
                loss.name, build_strategy=build_strategy)

        token = exe.run(compiled_program, feed=feed_dict, fetch_list=[tokens])
        if token == self.token:
            return False
        else:
            return True

    def _save_controller(self, program):
        fluid.save(program, self.save_controller)
