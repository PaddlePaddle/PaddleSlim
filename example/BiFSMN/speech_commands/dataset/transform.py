import paddle
import random
import math

class ChangeAmplitude(object):
    """Changes amplitude of an audio randomly."""

    def __init__(self, prop=0.5, amplitude_range=(0.7, 1.1)):
        self.amplitude_range = amplitude_range
        self.prop = prop

    def __call__(self, data: paddle.Tensor):
        if random.uniform(0, 1) <= self.prop:
            data = data * random.uniform(*self.amplitude_range)
        return data


class FixAudioLength(object):
    """Either pads or truncates an audio into a fixed length."""

    def __init__(self, time=1, sample_rate=16000):
        self.target_len = time * sample_rate

    def __call__(self, data: paddle.Tensor):
        cur_len = tuple(data.shape)[1]
        if self.target_len <= cur_len:
            data = data[:, :self.target_len]
        else:
            data = paddle.nn.functional.pad(x=data, pad=(0, self.target_len -
                cur_len), pad_from_left_axis=False)
        return data


class ChangeSpeedAndPitchAudio(object):
    """Change the speed of an audio. This transform also changes the pitch of the audio."""

    def __init__(self, prop=0.5, max_scale=0.2, sample_rate=16000):
        self.max_scale = max_scale
        self.sample_rate = sample_rate
        self.prop = prop

    def __call__(self, data):  
        if random.uniform(0, 1) <= self.prop:
            scale = random.uniform(-self.max_scale, self.max_scale)
            speed_fac = 1.0 / (1 + scale)
            _, channel= data.shape

            H = int(math.sqrt(channel))  
            W = channel // H
            data = data[:, :H * W]  
            data = data.reshape([1, H, W])  
            data = paddle.nn.functional.interpolate(x=data.unsqueeze(axis=1
                ), scale_factor=speed_fac, mode='nearest').squeeze(axis=1)
          
            H_new, W_new = data.shape[1], data.shape[2]
            data = data.reshape([1, H_new * W_new])  
            
        return data


class TimeshiftAudio(object):
    """Shifts an audio randomly."""

    def __init__(self, prop=0.5, max_shift_seconds=0.2, sample_rate=16000):
        self.shift_len = max_shift_seconds * sample_rate
        self.prop = prop

    def __call__(self, data):
        if random.uniform(0, 1) <= self.prop:
            shift = random.randint(-self.shift_len, self.shift_len)
            a = -min(0, shift)
            b = max(0, shift)
            data = paddle.nn.functional.pad(x=data, pad=(a, b), mode=
                'constant', pad_from_left_axis=False)
            data = data[:, :tuple(data.shape)[1] - a] if a else data[:, b:]
        return data
