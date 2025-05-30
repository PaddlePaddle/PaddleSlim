"""
TODO: add doc for module
"""
import paddle

__all__ = ["NetOption"]
"""
You can run your script with CUDA_VISIBLE_DEVICES=5,6 python your_script.py
or set the environment variable in the script by os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'
to map GPU 5, 6 to device_ids 0, 1, respectively.
"""
class NetOption(object):

    def __init__(self):
        #  ------------ General options ----------------------------------------
        self.save_path = ""  # log path
        self.dataPath = "/home/dataset/"  # path for loading data set
        self.dataset = "cifar10"  # options: imagenet | cifar10 | cifar100 | imagenet100 | mnist
        self.manualSeed = 1  # manually set RNG seed
        self.nGPU = 1  # number of GPUs to use by default
        self.GPU = 0  # default gpu to use, options: range(nGPU)

        # ------------- Data options -------------------------------------------
        self.nThreads = 4  # number of data loader threads

        # ------------- Training options ---------------------------------------
        self.testOnly = False  # run on validation set only
        self.tenCrop = False  # Ten-crop testing

        # ---------- Optimization options --------------------------------------
        self.nEpochs = 200  # number of total epochs to train
        self.batchSize = 128  # mini-batch size
        self.momentum = 0.9  # momentum
        self.weightDecay = 1e-4  # weight decay 1e-4
        self.opt_type = "SGD"

        self.lr = 0.1  # initial learning rate
        self.lrPolicy = "multi_step"  # options: multi_step | linear | exp | fixed
        self.power = 1  # power for learning rate policy (inv)
        self.step = [0.6, 0.8]  # step for linear or exp learning rate policy
        self.endlr = 0.001  # final learning rate, oly for "linear lrpolicy"
        self.decayRate = 0.1  # lr decay rate

        # ---------- Model options ---------------------------------------------
        self.netType = "PreResNet"  # options: ResNet | PreResNet | GreedyNet | NIN | LeNet5
        self.experimentID = "refator-test-01"
        self.depth = 20  # resnet depth: (n-2)%6==0
        self.nClasses = 10  # number of classes in the dataset
        self.wideFactor = 1  # wide factor for wide-resnet

        # ---------- Resume or Retrain options ---------------------------------------------
        self.retrain = None  # path to model to retrain with, load model state_dict only
        self.resume = None  # path to directory containing checkpoint, load state_dicts of model and optimizer, as well as training epoch

        # ---------- Visualization options -------------------------------------
        self.drawNetwork = True
        self.drawInterval = 30

        self.torch_version = paddle.__version__
        torch_version_split = self.torch_version.split("_")
        self.torch_version = torch_version_split[0]
        # check parameters
        # self.paramscheck()

    def paramscheck(self):
        if self.torch_version != "0.2.0":
            self.drawNetwork = False
            print("|===>DrawNetwork is supported by PyTorch with version: 0.2.0. The used version is ", self.torch_version)

        if self.netType in ["PreResNet", "ResNet"]:
            self.save_path = "log_%s%d_%s_bs%d_lr%0.3f_%s/" % (
                self.netType, self.depth, self.dataset,
                self.batchSize, self.lr, self.experimentID)
        else:

            self.save_path = "log_%s_%s_bs%d_lr%0.3f_%s/" % (
                self.netType, self.dataset,
                self.batchSize, self.lr, self.experimentID)

        if self.dataset in ["cifar10", "mnist"]:
            self.nClasses = 10
        elif self.dataset == "cifar100":
            self.nClasses = 100
        elif self.dataset == "imagenet" or "thi_imgnet":
            self.nClasses = 1000
        elif self.dataset == "imagenet100":
            self.nClasses = 100

        if self.depth >= 100:
            self.drawNetwork = False
            print("|===>draw network with depth over 100 layers, skip this step")
