class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class BaseQuantizationManager(metaclass=Singleton):
    def __init__(self, qconfig=None):
        self.qconfig = qconfig

    # def __enter__(self):
    #     self.enable()
    #     return self

    # def __exit__(self, *args):
    #     self.disable()

    @classmethod
    def build_quantization_manager(cls, qconfig):
        """Create specific quantization_manager"""
        return cls(qconfig)