import os
import paddle
from typing import Optional, Union
from pathlib import Path
import random
NOISE_FOLDER = '_background_noise_'
import os
from .dataset import SPEECHCOMMANDS as SpeechCommandsDataset
from .dataset import load_audio
import numpy as np
from utils.AmplitudeToDB import AmplitudeToDB
from paddle.audio.features import MelSpectrogram

class SpeechCommandV1(paddle.io.Dataset):
  
    def __init__(self, root: Union[str, Path], folder_in_archive: str=
        'SpeechCommands', download: bool=False, subset: Optional[str]=None,
        silence_percent=0.1, transform=None, num_classes=12, noise_ratio=
        None, noise_max_scale=0.4, cache_origin_data=False, version=
        'speech_commands_v0.02', config=None) ->None:
        self.classes = ['yes', 'no', 'up', 'down', 'left', 'right', 'on',
            'off', 'stop', 'go', 'zero', 'one', 'two', 'three', 'four',
            'five', 'six', 'seven', 'eight', 'nine', 'bed', 'bird', 'cat',
            'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow',
            'backward', 'forward', 'follow', 'learn', 'visual']
        self.classes_12 = ['unknown', 'silence', 'yes', 'no', 'up', 'down',
            'left', 'right', 'on', 'off', 'stop', 'go']
        self.classes_20 = ['yes', 'no', 'up', 'down', 'left', 'right', 'on',
            'off', 'stop', 'go', 'zero', 'one', 'two', 'three', 'four',
            'five', 'six', 'seven', 'eight', 'nine']
        self.classes_35 = ['backward', 'bed', 'bird', 'cat', 'dog', 'down',
            'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy',
            'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on',
            'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
            'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
        dataset = SpeechCommandsDataset(root, version,
            folder_in_archive, download, subset)
        data_path = os.path.join(root, folder_in_archive, version)
        self.num_classes = num_classes
        self.datas = list()
        for fileid in dataset._walker:
            relpath = os.path.relpath(fileid, data_path)
            label, _ = os.path.split(relpath)
            label = self.name_to_label(label)
            if label == -1:
                continue
            self.datas.append([fileid, label])
        self.sample_rate = 16000
        if silence_percent > 0 and num_classes == 12:
            silence_data = [['', self.name_to_label('silence')] for _ in
                range(int(len(dataset) * silence_percent))]
            self.datas.extend(silence_data)
        self.noise_folder = os.path.join(root, folder_in_archive, version,
            NOISE_FOLDER)
        self.noise_files = sorted(str(p) for p in Path(self.noise_folder).
            glob('*.wav')
            ) if subset == 'training' and noise_ratio != None else None
        self.transform = transform
        self.noise_ratio = noise_ratio
        self.noise_max_scale = noise_max_scale
        self.silence_ratio = silence_percent
        if noise_ratio is not None and subset is 'training':
            assert 0 < noise_max_scale < 1
        assert num_classes == 12 or num_classes == 20 or num_classes == 35, 'only support V1-12 now'
        self.cache_origin = cache_origin_data
        self.origin_datas = dict()
        self.origin_noise_datas = dict()
        self.transform2 = paddle.vision.transforms.Compose([
            AmplitudeToDB()
        ])
        self.config = config

    def __len__(self):
        return len(self.datas)

    def label_to_name(self, label):
        if label >= 12:
            return 'unknown'
        return self.classes_12[label]

    def name_to_label(self, name):
        if self.num_classes == 12:
            if name not in self.classes_12:
                return 0
            return self.classes_12.index(name)
        elif self.num_classes == 20:
            if name not in self.classes_20:
                return 0 if self.classes_20 == 'unknown' else -1
            return self.classes_20.index(name)
        elif self.num_classes == 35:
            if name not in self.classes_35:
                return 0 if self.classes_35 == 'unknown' else -1
            return self.classes_35.index(name)
        else:
            raise RuntimeError

    def __getitem__(self, index):
        """
        return feature and label
        """
        if index in self.origin_datas.keys():
            [waveform, _, label] = self.origin_datas[index]
        else:
            waveform, sample_rate, label = self.pull_origin(index)
            if sample_rate != self.sample_rate:
                raise RuntimeError
            if self.cache_origin:
                self.origin_datas[index] = [waveform, sample_rate, label]
        if self.noise_files is not None and random.uniform(0, 1
            ) < self.noise_ratio:
            noise_file = random.choice(self.noise_files)
            if noise_file in self.origin_noise_datas.keys():
                waveform_noise = self.origin_noise_datas[noise_file]
            else:
                waveform_noise, _ = load_audio(noise_file)
                if self.cache_origin:
                    self.origin_noise_datas[noise_file] = waveform_noise
            noise_len = tuple(waveform_noise.shape)[1]
            wav_len = tuple(waveform.shape)[1]
            if noise_len >= wav_len:
                rand_start = random.randint(0, noise_len - wav_len - 1)
                waveform_noise = waveform_noise[:, rand_start:wav_len +
                    rand_start]
            else:
                waveform_noise = paddle.nn.functional.pad(x=waveform_noise,
                    pad=(0, wav_len - noise_len), pad_from_left_axis=False)
            random_scale = random.uniform(0, self.noise_max_scale)
            waveform = waveform * (1 - random_scale
                ) + waveform_noise * random_scale
        if self.transform is not None:
            # print(f"before transform: {waveform.shape}")
            waveform = self.transform(waveform)
            feature_extractor = MelSpectrogram(sr=16000, n_fft=2048, hop_length=512, n_mels=self.config.n_mels)
            waveform = feature_extractor(waveform)
            waveform = self.transform2(waveform)
            # print(f"after transform: {waveform.shape}")
        return waveform, label

    def pull_origin(self, index):
        """
        get original item
        """
        [data_id, label] = self.datas[index]
        if data_id != '':
            waveform, sample_rate = load_audio(data_id)
        else:
            waveform = paddle.zeros(shape=[1, 16000])
            sample_rate = 16000
        return waveform, sample_rate, label
