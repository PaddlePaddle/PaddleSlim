class Model(fluid.dygraph.Layer):
    #@ofa_supernet(kernel_size=(3,5,7), expand_ratio=(1, 2, 4))
    def __init__(self):
        super(Model, self).__init__()
        #with ofa_supernet(
        #        kernel_size=(3, 5, 7), expand_ratio=(1, 2, 4)) as ofa_super:
        with ofa_supernet(
                kernel_size=(3, 5, 7),
                channel=((4, 8, 12), (8, 12, 16), (8, 12, 16))) as ofa_super:
            models = []
            models += [nn.Conv2D(3, 4, 3)]
            models += [nn.InstanceNorm(4)]
            models += [nn.ReLU()]
            models += [nn.Conv2DTranspose(4, 4, 3, groups=4, use_cudnn=True)]
            models += [nn.BatchNorm(4)]
            models += [nn.ReLU()]
            models += [
                fluid.dygraph.Pool2D(
                    pool_type='avg', global_pooling=True, use_cudnn=False)
            ]
            models += [nn.Linear(4, 3)]
            models += [nn.ReLU()]
            models = ofa_super.convert(models)
        self.models = nn.Sequential(*models)

    def forward(self, inputs):
        for idx, layer in enumerate(self.models):
            if idx == (len(self.models) - 2):
                inputs = fluid.layers.reshape(
                    inputs, shape=[inputs.shape[0], -1])
            inputs = layer(inputs)
        return inputs
        #return self.models(inputs)


def test_convert():
    import numpy as np
    data_np = np.random.random((1, 3, 10, 10)).astype('float32')
    fluid.enable_dygraph()
    net = Model()
    ofa_model = OFA(net)
    adam = fluid.optimizer.Adam(
        learning_rate=0.001, parameter_list=ofa_model.parameters())

    for name, sublayer in net.named_sublayers():
        print(name, sublayer)
        if getattr(sublayer, '_filter_size', None) != None and getattr(
                sublayer, '_num_filters', None) != None:
            print(name, sublayer._num_channels, sublayer._num_filters,
                  sublayer._filter_size)
        if getattr(sublayer, 'candidate_config', None) != None:
            print(name, sublayer.candidate_config)

    data = fluid.dygraph.to_variable(data_np)
    for _ in range(10):
        out = ofa_model(data)
        loss = fluid.layers.reduce_mean(out)
        print(loss.numpy())
        loss.backward()
        adam.minimize(loss)
        adam.clear_gradients()


def test_ofa():
    # NOTE: case 1
    data_np = np.random.random((1, 3, 10, 10)).astype(np.float32)
    label_np = np.random.random((1)).astype(np.float32)

    default_run_config = {
        'train_batch_size': 256,
        'eval_batch_size': 64,
        'n_epochs': [[1], [2, 3]],
        'init_learning_rate': [[0.001], [0.0001, 0.003]],
        'dynamic_batch_size': [1, 1],
        'total_images': 1281167
    }
    run_config = RunConfig(**default_run_config)

    assert len(run_config.n_epochs) == len(run_config.dynamic_batch_size)
    assert len(run_config.n_epochs) == len(run_config.init_learning_rate)

    default_distill_config = {
        'lambda_distill': 0.01,
        'teacher_model': SuperNet,
        'mapping_layers': ['models.4.fn']
    }
    distill_config = DistillConfig(**default_distill_config)

    fluid.enable_dygraph()
    model = SuperNet()
    ofa_model = OFA(model, run_config, distill_config=distill_config)
    print(ofa_model.state_dict().keys())
    #for name, sublayer in ofa_model.named_sublayers():
    #    print(name, sublayer)

    data = fluid.dygraph.to_variable(data_np)
    label = fluid.dygraph.to_variable(label_np)

    start_epoch = 0
    for idx in range(len(run_config.n_epochs)):
        cur_idx = run_config.n_epochs[idx]
        for ph_idx in range(len(cur_idx)):
            cur_lr = run_config.init_learning_rate[idx][ph_idx]
            adam = fluid.optimizer.Adam(
                learning_rate=cur_lr, parameter_list=ofa_model.parameters())
            for epoch_id in range(start_epoch,
                                  run_config.n_epochs[idx][ph_idx]):
                # add for data in dataset:
                for model_no in range(run_config.dynamic_batch_size[idx]):
                    output, _ = ofa_model(data)
                    loss = fluid.layers.reduce_mean(output)
                    dis_loss = ofa_model.calc_distill_loss()
                    print('epoch: {}, loss: {}, distill loss: {}'.format(
                        epoch_id, loss.numpy()[0], dis_loss.numpy()[0]))
                    loss.backward()
                    adam.minimize(loss)
                    adam.clear_gradients()
            start_epoch = run_config.n_epochs[idx][ph_idx]


# NOTE: case 2
#class BaseManager:
#    def __init__(self, model, run_config, optim_fn, train_dataset, eval_dataset):
#        self.model = model
#        self.run_config = run_config
#        self.loss = loss_fn
#        ### optim_fn = dict('Adam': parameter_list: PARAM, learning_rate) ??? 
#        self.optim = optim_fn
#        self.train_dataset = train_dataset
#        self.eval_dataset = eval_dataset
#
#    def train_one_epoch(self):
#        for image, label in self.train_dataset():
#            out = self.model(image)
#            loss = self.loss(out, label)
#            self.optim.clear_gradient()
#            loss.backward()
#            self.optim.minimize(loss)
#
#    def eval_one_epoch(self):
#        for image, label in self.eval_dataset():
#            out = self.model(image)
#            acc_top1, acc_top5 = accuracy(out, label)
#
#        # compute final acc
#
#if name == '__main__':
#    data_np = np.random.random((1, 3, 10, 10)).astype(np.float32)
#    label_np = np.random.random((1)).astype(np.float32)
#
#    fluid.enable_dygraph()
#    model = SuperNet()
#    run_config = dict(#TODO)
#    my_manager = BaseManager(model, run_config)
#    ofa_model = OFA(my_manager)
#    ofa_mode.train()
