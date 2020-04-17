import paddle.fluid as fluid
from paddleslim.teachers.bert.reader.cls import *
from paddleslim.nas.darts.search_space import AdaBERTClassifier
from paddleslim.nas.darts import DARTSearch


def main():
    place = fluid.CUDAPlace(0)

    BERT_BASE_PATH = "./data/pretrained_models/uncased_L-12_H-768_A-12/"
    bert_config_path = BERT_BASE_PATH + "/bert_config.json"
    vocab_path = BERT_BASE_PATH + "/vocab.txt"
    data_dir = "./data/glue_data/MNLI/"
    max_seq_len = 512
    do_lower_case = True
    batch_size = 32
    epoch = 30

    processor = MnliProcessor(
        data_dir=data_dir,
        vocab_path=vocab_path,
        max_seq_len=max_seq_len,
        do_lower_case=do_lower_case,
        in_tokens=False)

    train_reader = processor.data_generator(
        batch_size=batch_size,
        phase='train',
        epoch=epoch,
        dev_count=1,
        shuffle=True)

    val_reader = processor.data_generator(
        batch_size=batch_size,
        phase='train',
        epoch=epoch,
        dev_count=1,
        shuffle=True)

    with fluid.dygraph.guard(place):
        model = AdaBERTClassifier(
            3,
            teacher_model="/work/PaddleSlim/demo/bert_1/checkpoints/steps_23000"
        )
        searcher = DARTSearch(
            model,
            train_reader,
            val_reader,
            batchsize=batch_size,
            num_epochs=epoch,
            log_freq=10)
        searcher.train()


if __name__ == '__main__':
    main()
