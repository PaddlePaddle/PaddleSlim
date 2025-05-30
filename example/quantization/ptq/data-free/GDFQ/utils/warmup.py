from torchlearning.mio import MIO

train_dataset = MIO("/home/datasets/imagenet_mio/train/")
test_dataset = MIO("/home/datasets/imagenet_mio/val/")

for i in range(train_dataset.size):
    print(i)
    train_dataset.fetchone(i)
for i in range(test_dataset.size):
    print(i)
    test_dataset.fetchone(i)