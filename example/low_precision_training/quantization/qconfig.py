class QParam(object):
    def __init__(
        self, 
        qtype: str = 'fp',
        bitwidth: int = 32,
        granularity: str = 'per-layer',
        scale = None
    ):
        self.qtype = qtype
        self.bitwidth = bitwidth
        self.granularity = granularity
        self.scale = None

class QConfig(object):
    
    def __init__(self):
        self.forward_weight = None
        self.forward_input = None
        self.backward_weight = None
        self.backward_grad_x_weight = None
        self.backward_input = None
        self.backward_grad_x_input = None