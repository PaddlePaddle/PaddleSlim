import paddle.fluid as fluid
from paddleslim.nas.darts.search_space import ConvBERTClassifier

place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)

with fluid.dygraph.guard(place):

    bert = ConvBERTClassifier(3)
    bert.fit("./data/glue_data/MNLI/",
             5,
             batch_size=16,
             use_data_parallel=False,
             learning_rate=0.00005,
             save_steps=1000)
