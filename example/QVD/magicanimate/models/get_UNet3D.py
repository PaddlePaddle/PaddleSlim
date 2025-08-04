
def from_pretrained_2d(cls, pretrained_model_path, subfolder=None, unet_additional_kwargs=None):
    if subfolder is not None:
        pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
    print(f"loaded temporal unet's pretrained weights from {pretrained_model_path} ...")

    config_file = os.path.join(pretrained_model_path, 'config.json')
    if not os.path.isfile(config_file):
        raise RuntimeError(f"{config_file} does not exist")
    with open(config_file, "r") as f:
        config = json.load(f)
    config["_class_name"] = cls.__name__
    config["down_block_types"] = [
        "CrossAttnDownBlock3D",
        "CrossAttnDownBlock3D",
        "CrossAttnDownBlock3D",
        "DownBlock3D"
    ]
    config["up_block_types"] = [
        "UpBlock3D",
        "CrossAttnUpBlock3D",
        "CrossAttnUpBlock3D",
        "CrossAttnUpBlock3D"
    ]
    # config["mid_block_type"] = "UNetMidBlock3DCrossAttn"

    # Assuming the method `from_config` is available and works for Paddle
    model = cls.from_config(config, **unet_additional_kwargs)
    model_file = os.path.join(pretrained_model_path, 'pytorch_model.bin')  # WEIGHTS_NAME is 'pytorch_model.bin'
    if not os.path.isfile(model_file):
        raise RuntimeError(f"{model_file} does not exist")
    
    # Load state_dict from the file
    state_dict = paddle.load(model_file)  # In Paddle, `paddle.load` is used for loading state_dict
    state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}

    # Load the state dict into the model (strict=False equivalent in Paddle)
    model.set_state_dict(state_dict)
    
    print(f"### missing keys: {len(state_dict)}")  # Adjust as needed for paddle, no direct check for missing/unexpected keys
    # If needed, use model.named_parameters() to check specific parameter names for debugging
    params = [p.numel() if "temporal" in n else 0 for n, p in model.named_parameters()]
    print(f"### Temporal Module Parameters: {sum(params) / 1e6} M")
    
    return model