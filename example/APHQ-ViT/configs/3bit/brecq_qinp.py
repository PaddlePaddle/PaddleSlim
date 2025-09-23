
class Config:
    def __init__(self):
        # calibration settings
        self.optim_size = 1024
        self.calib_size = 128
        self.optim_batch_size = 32
        self.calib_batch_size = 32
        self.w_bit = 3
        self.a_bit = 3
        self.qconv_a_bit = 8
        self.qhead_a_bit = 3
        self.calib_metric = 'mse'
        self.matmul_head_channel_wise = True
        self.token_channel_wise = False
        self.eq_n = 128
        self.search_round = 3
        # optimization settings
        self.keep_gpu = True
        self.optim_metric = 'mse'
        self.use_mean_hessian = False
        self.temp = 20
        # reconstruction settings
        self.recon_metric = 'hessian_perturb'
        self.pct = 0.99
        # qdrop settings
        self.optim_mode = 'qinp'
        self.drop_prob = 1.0
