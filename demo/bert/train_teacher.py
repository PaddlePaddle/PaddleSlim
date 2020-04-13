import paddle.fluid as fluid
from paddleslim.teachers.bert import BERTClassifier

place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)

with fluid.dygraph.guard(place):

    bert = BERTClassifier(3)
    bert.fit("./data/glue_data/MNLI/", 1, batch_size=32)
