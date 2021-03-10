import random
import paddle


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for img in images:
            img = img.detach()
            img = paddle.unsqueeze(img, axis=0)
            if self.num_imgs < self.pool_size:
                self.images.append(img)
                return_images.append(img)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    temp = self.images[random_id]
                    self.images[random_id] = img
                    return_images.append(temp)
                else:
                    return_images.append(img)

        return_images = paddle.concat(return_images, axis=0)
        return return_images
