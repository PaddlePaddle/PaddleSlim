import paddle

class DummyQuantizer:
    def __call__(self, tensor):
        return tensor

    def __repr__(self):
        return 'DummyQuantizer - fp32'

    def calculate_quantization_scale(self, tensor, max_iter=50):
        # PaddlePaddle equivalent to torch.tensor(1.0).to(tensor.device)
        return paddle.to_tensor(1.0, place=tensor.place)

    def set_scale(self, scale):
        self.scale = scale


def dummy_quantizer(*args, **kwargs):
    return DummyQuantizer()
    
def full_quantizer(*args, **kwargs):
    return DummyQuantizer()