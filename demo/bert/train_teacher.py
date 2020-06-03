import paddle.fluid as fluid
from paddleslim.teachers.bert import BERTClassifier

place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)

with fluid.dygraph.guard(place):

    bert = BERTClassifier(3)
    bert.fit("./data/glue_data/MNLI/",
             5,
             batch_size=32,
             use_data_parallel=True,
             learning_rate=0.00005,
             save_steps=1000)
