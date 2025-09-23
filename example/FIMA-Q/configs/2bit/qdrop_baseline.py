
class Config:
    def __init__(self):
        # calibration settings
        self.optim_size = 1024
        self.calib_size = 128
        self.optim_batch_size = 32
        self.calib_batch_size = 32
        self.w_bit = 2
        self.a_bit = 2
        self.qconv_a_bit = 8
        self.qhead_a_bit = 2
        self.calib_metric = 'mse'
        self.matmul_head_channel_wise = True
        self.token_channel_wise = True
        self.eq_n = 128
        self.search_round = 3
        # optimization settings
        self.keep_gpu = True
        self.optim_metric = 'mse'
        self.use_mean_hessian = True
        self.temp = 20
        # reconstruction settings
        self.recon_metric = 'hessian_perturb'
        self.pct = 0.98
        # qdrop settings
        self.optim_mode = 'qdrop'
        self.drop_prob = 0.5
