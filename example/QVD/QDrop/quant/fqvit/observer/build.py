from .minmax import MinmaxObserver
from .omse import OmseObserver
from .ema import EmaObserver
from .percentile import PercentileObserver
from .pts import PtsObserver
from .twin import TwinObserver
str2observer = {'minmax': MinmaxObserver, 'ema': EmaObserver, 'omse':
    OmseObserver, 'percentile': PercentileObserver, 'pts': PtsObserver,
    'twin': TwinObserver}


def build_observer(observer_str, module_type, bit_type, calibration_mode):
    observer = str2observer[observer_str]
    return observer(module_type, bit_type, calibration_mode)
