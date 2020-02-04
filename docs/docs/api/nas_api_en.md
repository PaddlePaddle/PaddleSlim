## How to configure the search space:
configure the search space by parameter. More search space can reference [search_space](../search_space_en.md)

**Args:**

- **input_size(int|None)**: -`input_size` represents the input size of the feature map. `input_size` and `output_size` can be use in compute the times of downsample in network.
- **output_size(int|None)**: -`output_size` represents the output size of the feature map. `input_size` and `output_size` can be use in compute the times of downsample in network.
- **block_num(int|None)**: -`block_num` represents the number of block in search space. A Block consists by one or more operators, each block is similar, one or more block constitute a network.
- **block_mask(list|None)**: - `block_mask` is a list consists by 0 and 1. 0 represents normal block, 1 represents reduction block. Reduction block means after this block, the size of feature map will reduce to half. Normal block means after this block, the size of feature map will not change. If `block_mask` is not None, the network composed according to `block_mask`, `input_size`,`output_size` and `block_num` is invalid.

## SANAS
paddleslim.nas.SANAS(configs, server_addr=("", 8881), init_temperature=None, reduce_rate=0.85, init_tokens=None, search_steps=300, save_checkpoint='./nas_checkpoint', load_checkpoint=None, is_server=True)[source](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/nas/sa_nas.py#L36)
: SANAS(Simulated Annealing Neural Architecture Search) is a neural architecture search algorithm based on simulated annealing, used in discrete search task generally.

**Args:**

- **configs(list<tuple>)** - config lists of search space, in the form of `[(key, {input_size, output_size, block_num, block_mask})]` or `[(key)]` (MobileNetV2, MobileNetV1 and ResNet uses origin network as search space, so only set key is enough), `input_size` and `output_size` represent input size and output size of feature map. `block_num` represents the number of block in search space. `block_mask` is a list consists by 0 and 1. 0 represents normal block, 1 represents reduction block. More search space can reference [search_space](../search_space_en.md)
- **server_addr(tuple)** - server address, including ip and port of server. If ip is None or "", will use host ip if `is_server = True`. Default: ("", 8881).
- **init_temperature(float)** - initial temperature in SANAS. If `init_temperature` and `init_tokens` are None, default initial temperature is 10.0, if `init_temperature` is None and `init_tokens` is not None, default initial temperature is 1.0. The detail configuration about the `init_temperature`  please reference Note. Default: None.
- **reduce_rate(float)** - reduce rate in SANAS. The detail configuration about the `reduce_rate` please reference Note. Default: 0.85.
- **init_tokens(list|None)** - initial token. If `init_tokens` is None, SANAS will random generate initial tokens. Default: None.
- **search_steps(int)** - iterations in search. Default: 300.
- **save_checkpoint(str|None)** - directory to save checkpoint. If `save_checkpoint` is None, it means don't save checkpoint. Default: `./nas_checkpoint`.
- **load_checkpoint(str|None)** - directory to load checkpoint. If `load_checkpoint` is None, it means don't load checkpoint. Default: None.
- **is_server(bool)** - whether to start a server. None: True.

**Return:**
a instance of class SANAS.

**Example:**
```python
from paddleslim.nas import SANAS
config = [('MobileNetV2Space')]
sanas = SANAS(configs=config)
```

!!! note "Note"
  - Why need to set initial temperature and reduce rate:<br>
    - SA algorithm preserve a base token(initial token is the first base token, can be set by yourself or random generate) and base score(initial score is -1), next token will be generated based on base token. During the search, if the score which is obtained by the model corresponding to the token is greater than the score which is saved in SA corresponding to base token, current token saved as base token certainly; if score which is obtained by the model corresponding to the token is less than the score which is saved in SA correspinding to base token, current token saved as base token with a certain probability.<br>
    - For initial temperature, higher is more unstable, it means that SA has a strong possibility to save current token as base token if current score is smaller than base score saved in SA.
    - For initial temperature, lower is more stable, it means that SA has a small possibility to save current token as base token if current score is smaller than base score saved in SA.<br>
    - For reduce rate, higher means SA algorithm has slower convergence.<br>
    - For reduce rate, lower means SA algorithm has faster convergence.<br>

  - How to set initial temperature and reduce rate:<br>
    - If there is a better initial token, and want to search based on this token,  we suggest start search experiment in the steady state of the SA algorithm, initial temperature can be set to a small value, such as 1.0, and reduce rate can be set to a large value, such as 0.85. If you want to start search experiment based on the better token with greedy algorithm, which only saved current token as base token if current score higher than base score saved in SA algorithm, reduce rate can be set to a extremely small value, such as 0.85 ** 10.<br>
    - If initial token is generated randomly, it means initial token is a worse token, we suggest start search experiment in the unstable state of the SA algorithm, explore all random tokens as much as possible, and get a better token. Initial temperature can be set a higher value, such as 1000.0, and reduce rate can be set to a small value.


paddleslim.nas.SANAS.next_archs()
: get next model architecture.

**Return:**
list of model architecture instance.

**Example:**
```python
import paddle.fluid as fluid
from paddleslim.nas import SANAS
config = [('MobileNetV2Space')]
sanas = SANAS(configs=config)
input = fluid.data(name='input', shape=[None, 3, 32, 32], dtype='float32')
archs = sanas.next_archs()
for arch in archs:
    output = arch(input)
    input = output
print(output)
```

paddleslim.nas.SANAS.reward(score)
: send the score of this model architecture to SANAS.

**Args:**

- **score<float>:** - score of this model architecture, bigger is better.

**Return:**
return model architecture update succeed or failed. Update succeed if return True, failed if return False.

**Example:**
```python
import paddle.fluid as fluid
from paddleslim.nas import SANAS
config = [('MobileNetV2Space')]
sanas = SANAS(configs=config)
archs = sanas.next_archs()

### assume the score is 1, request a real score in practical.
score=float(1.0)
sanas.reward(float(score))
```

paddlesim.nas.SANAS.tokens2arch(tokens)
: get corresponding model by tokens, use final tokens to train final experiment usually. `tokens` is a list, a list of token corresponds a model architecture.

**Args:**

- **tokens(list):** - a list of token. The length and range based on search space.

**Return:**
a model architecture instance according to tokens.

**Example:**
```python
import paddle.fluid as fluid
from paddleslim.nas import SANAS
config = [('MobileNetV2Space')]
sanas = SANAS(configs=config)
input = fluid.data(name='input', shape=[None, 3, 32, 32], dtype='float32')
tokens = ([0] * 25)
archs = sanas.tokens2arch(tokens)[0]
print(archs(input))
```

paddleslim.nas.SANAS.current_info()
: get current token and best token and reward in search.

**Return:**
A dictionary including current token, best token and reward in search.

**Example:**
```python
import paddle.fluid as fluid
from paddleslim.nas import SANAS
config = [('MobileNetV2Space')]
sanas = SANAS(configs=config)
print(sanas.current_info())
```