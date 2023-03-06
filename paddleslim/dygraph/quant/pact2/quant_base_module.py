import copy

###########################################################
# default settings for quantization

class ConstrainBiasType:
    CONSTRAIN_BIAS_TYPE_NONE = 0
    CONSTRAIN_BIAS_TYPE_SATURATE = 1
    CONSTRAIN_BIAS_TYPE_REDUCE_WEIGHT_SCALE = 2
    CONSTRAIN_BIAS_TYPE_REDUCE_FEATURE_SCALE = 3


class QuantDefaults:
    RANGE_SHRINK_WEIGHTS_DEFAULT = 0.0
    POWER2_WEIGHT_RANGE_DEFAULT = True
    POWER2_ACTIVATION_RANGE_DEFAULT = True
    CONSTRAIN_BIAS_DEFAULT = ConstrainBiasType.CONSTRAIN_BIAS_TYPE_SATURATE
