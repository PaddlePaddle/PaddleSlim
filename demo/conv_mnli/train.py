import time
import paddle.fluid as fluid
from paddleslim.teachers.bert.reader.cls import *
from model.cls import ClsModelLayer


def main():
    place = fluid.CUDAPlace(0)

    BERT_BASE_PATH = "./data/pretrained_models/uncased_L-12_H-768_A-12/"
    bert_config_path = BERT_BASE_PATH + "/bert_config.json"
    vocab_path = BERT_BASE_PATH + "/vocab.txt"
    data_dir = "./data/glue_data/MNLI/"
    max_seq_len = 128
    do_lower_case = True
    batch_size = 128
    epoch = 600

    processor = MnliProcessor(
        data_dir=data_dir,
        vocab_path=vocab_path,
        max_seq_len=max_seq_len,
        do_lower_case=do_lower_case,
        in_tokens=False)

    train_reader = processor.data_generator(
        batch_size=batch_size,
        phase='train',
        epoch=1,
        dev_count=1,
        shuffle=True)

    val_reader = processor.data_generator(
        batch_size=batch_size,
        phase='dev',
        epoch=1,
        dev_count=1,
        shuffle=False)

    with fluid.dygraph.guard(place):
        model = ClsModelLayer(3)
        optimizer = fluid.optimizer.MomentumOptimizer(
            0.001,
            0.9,
            regularization=fluid.regularizer.L2DecayRegularizer(3e-4),
            parameter_list=model.parameters())

        for i in range(epoch):
            model.train()
            losses = []
            accs = []
            start = time.time()
            for step_id, data in enumerate(train_reader()):
                loss, acc, num = model(data)
                loss.backward()
                optimizer.minimize(loss)
                model.clear_gradients()
                losses.append(loss.numpy())
                accs.append(acc.numpy())
                if step_id % 50 == 0:
                    time_cost = time.time() - start
                    start = time.time()
                    speed = time_cost / 50.0
                    print(
                        "Train iter-[{}]-[{}] - loss: {:.4f}; acc: {:.4f}; speed: {:.3f}s/step; time: {}".
                        format(i, step_id,
                               np.mean(losses),
                               np.mean(accs), speed,
                               time.asctime(time.localtime())))
                    losses = []
                    accs = []

            model.eval()
            losses = []
            accs = []
            for step_id, data in enumerate(val_reader()):
                loss, acc, num = model(data)
                losses.append(loss.numpy())
                accs.append(acc.numpy())
            print("Eval epoch [{}]- loss: {:.4f}; acc: {:.4f}".format(
                i, np.mean(losses), np.mean(accs)))


if __name__ == '__main__':
    main()
