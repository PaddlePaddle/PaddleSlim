import paddle.fluid as fluid
import paddleslim.quant as quant
import unittest


class TestQuantEmbedding(unittest.TestCase):
    def test_quant_embedding(self):
        train_program = fluid.Program()
        with fluid.program_guard(train_program):
            input_word = fluid.data(
                name="input_word", shape=[None, 1], dtype='int64')
            input_emb = fluid.embedding(
                input=input_word,
                is_sparse=False,
                size=[100, 128],
                param_attr=fluid.ParamAttr(
                    name='emb',
                    initializer=fluid.initializer.Uniform(-0.005, 0.005)))

        infer_program = train_program.clone(for_test=True)

        use_gpu = True
        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        config = {'params_name': 'emb', 'quantize_type': 'abs_max'}
        quant_program = quant.quant_embedding(infer_program, place, config)


if __name__ == '__main__':
    unittest.main()
