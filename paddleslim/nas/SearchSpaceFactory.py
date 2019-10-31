from MobileNetV2Space import MobileNetV2Space

class SearchSpaceFactory(object):
    def __init__(self, key, config):
        self.key = key
        self.input_size = config['input_size']
        self.output_size = config['output_size']
        self.block_num = config['block_num']

    def get_search_space(self):
        if self.key == 'MobileNetV2':
            space = MobileNetV2Space(self.input_size, self.output_size, self.block_num)

        return space


import paddle.fluid as fluid
if __name__ == '__main__':
    config = {'input_size': 224, 'output_size': 7, 'block_num': 5}
    space = SearchSpaceFactory('MobileNetV2', config=config)
    
    my_space = space.get_search_space()
    model_arch = my_space.token2arch()
    
    train_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        model_input = fluid.layers.data(name='model_in', shape=[1, 3, 224, 224], dtype='float32', append_batch_size=False)
        predict = model_arch(model_input)
        print('output shape', predict.shape)
        

    #for op in train_prog.global_block().ops:
    #    print(op.type)
