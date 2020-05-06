import numpy as np
import scipy.misc

import paddle
from paddle import fluid

class LFW(object):
    def __init__(self, imgl, imgr):

        self.imgl_list = imgl
        self.imgr_list = imgr
        self.shuffle_idx = [i for i in range(len(self.imgl_list))]

    def reader(self):
        while True:
            if len(self.shuffle_idx) == 0:
                self.shuffle_idx = [i for i in range(len(self.imgl_list))]
                return
            index = self.shuffle_idx.pop(0)
            
            imgl = scipy.misc.imread(self.imgl_list[index])
            if len(imgl.shape) == 2:
                imgl = np.stack([imgl] * 3, 2)
            imgr = scipy.misc.imread(self.imgr_list[index])
            if len(imgr.shape) == 2:
                imgr = np.stack([imgr] * 3, 2)

            imglist = [imgl, imgl[:, ::-1, :], imgr, imgr[:, ::-1, :]]
            for i in range(len(imglist)):
                imglist[i] = (imglist[i] - 127.5) / 128.0
                imglist[i] = imglist[i].transpose(2, 0, 1)

            imgs = [img.astype('float32') for img in imglist]
            yield imgs

    def __len__(self):
        return len(self.imgl_list)


if __name__ == '__main__':
    pass