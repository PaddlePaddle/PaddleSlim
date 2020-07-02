import paddle.fluid as fluid
from paddleslim.teachers.bert import BERTClassifier
import sys

if __name__ == "__main__":
    k = int(sys.argv[1])
    count = int(sys.argv[2])
    outfile = sys.argv[3]
    start = count * (k - 1)
    end = start + count
    place = fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        teacher = BERTClassifier(3, model_path="./teacher_model/steps_23000")
        teacher.test("./data/glue_data/MNLI/", max_seq_len=128, batch_size=256)
        reader = teacher.cache(
            "./data/glue_data/MNLI/", outfile=outfile, start=start, end=end)
    #    reader = teacher.cache_reader("./teacher.data")
    #    for a, b, label, logit, loss in reader():
    #        print("a_ids: {}; b_ids: {}; label: {}; logit: {}; loss: {}".format(a.shape, b.shape, label.shape, logit.shape, loss.shape)) 
