import os
import logging
import time
import glob

import numpy as np
import tqdm
import paddle
import paddle.io as data

# import sys
# sys.path.append('./')
from models.diffusion_binarydm import Model
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path

import numpy as np
from PIL import Image


def paddle2hwcuint8(x, clip=False):
    if clip:
        x = paddle.clip(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif beta_schedule == "cosine":
        betas = betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = paddle.get_device()
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = paddle.to_tensor(betas).astype('float32')
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(0)
        alphas_cumprod_prev = paddle.concat(
            [paddle.ones([1]), alphas_cumprod[:-1]], 0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # paddle.concat(
            # [posterior_variance[1:2], betas[1:]], 0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clip(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        vdl_logger = self.config.vdl_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            use_shared_memory=False,
        )
        model = Model(config)

        model = model
        model = paddle.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = paddle.load(os.path.join(self.args.log_path, "ckpt.pdl"))
            model.set_state_dict({k.split("$model_")[-1]: v for k, v in states.items() if "$model_" in k})

            optimizer.set_state_dict({k.split("$optimizer_")[-1]: v for k, v in states.items() if "$optimizer_" in k})
            optimizer._epsilon = self.config.optim.eps
            start_epoch = states["$epoch"]
            step = states["$step"]
            if self.config.model.ema:
                ema_helper.set_state_dict({k.split("$ema_")[-1]: v for k, v in states.items() if "$ema_" in k})

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.shape[0]
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = data_transform(self.config, x)
                e = paddle.randn(x.shape)
                b = self.betas

                # antithetic sampling
                t = paddle.randint(
                    low=0, high=self.num_timesteps, shape=(n // 2 + 1,)
                )
                t = paddle.concat([t, self.num_timesteps - t - 1], 0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b)

                vdl_logger.add_scalar("loss", loss, step=step)

                logging.info(
                    f"step: {step}, loss: {loss.numpy()}, data time: {data_time / (i+1)}"
                )

                optimizer.clear_grad()
                loss.backward()
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = dict(
                        **{"$model_"+k: v for k, v in model.state_dict().items()},
                        **{"$optimizer_"+k: v for k, v in optimizer.state_dict().items()},
                        **{"$epoch": epoch},
                        **{"$step": step},
                    )
                    if self.config.model.ema:
                        states.update({"$ema_"+k: v for k, v in ema_helper.state_dict().items()})

                    paddle.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pdl".format(step)),
                    )
                    paddle.save(states, os.path.join(self.args.log_path, "ckpt.pdl"))

                data_start = time.time()

    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = paddle.load(
                    os.path.join(self.args.log_path, "ckpt.pdl")
                )
            else:
                states = paddle.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pdl"
                    )
                )
            model = model
            model = paddle.DataParallel(model)
            model.set_state_dict({k.split("$model_")[-1]: v for k, v in states.items() if "$model_" in k})

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.set_state_dict({k.split("$ema_")[-1]: v for k, v in states.items() if "$ema_" in k})
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.set_state_dict(paddle.load(ckpt))
            model = paddle.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with paddle.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = paddle.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                )

                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    Image.fromarray(np.uint8(x[i].numpy().transpose([1,2,0])*255)).save(
                        os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_sequence(self, model):
        config = self.config

        x = paddle.randn([
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
        ])

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with paddle.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].shape[0]):
                Image.fromarray(np.uint8(x[i][j].numpy().transpose([1,2,0])*255)).save(
                    os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = paddle.acos(paddle.sum(z1 * z2) / (paddle.norm(z1) * paddle.norm(z2)))
            return (
                paddle.sin((1 - alpha) * theta) / paddle.sin(theta) * z1
                + paddle.sin(alpha * theta) / paddle.sin(theta) * z2
            )

        z1 = paddle.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
        )
        z2 = paddle.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
        )
        alpha = paddle.arange(0.0, 1.01, 0.1)
        z_ = []
        for i in range(alpha.shape[0]):
            z_.append(slerp(z1, z2, alpha[i]))

        x = paddle.concat(z_, 0)
        xs = []

        # Hard coded here, modify to your preferences
        with paddle.no_grad():
            for i in range(0, x.shape[0], 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, paddle.concat(xs, 0))
        for i in range(x.shape[0]):
            Image.fromarray(np.uint8(x[i].numpy().transpose([1,2,0])*255)).save(os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass
