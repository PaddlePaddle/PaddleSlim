def encode_config(config):
    return '_'.join([str(c) for c in config['channels']])


def decode_config(config_str):
    channels = config_str.split('_')
    assert len(channels) == 6 or len(channels) == 8
    channels = [int(c) for c in channels]
    return {'channels': channels}
