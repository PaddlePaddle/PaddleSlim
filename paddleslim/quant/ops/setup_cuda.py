from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name='custom_setup_ops',
    ext_modules=CUDAExtension(
        sources=[
            "quantize_blockwise.cu",
            "dequantize_blockwise.cu"]
    )
)
