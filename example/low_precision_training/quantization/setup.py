# from setuptools import setup, Extension
# from torch.utils.cpp_extension import (CUDAExtension, CppExtension, BuildExtension)

# setup(name='cudnn_convolution',
#       ext_modules=[CUDAExtension('cudnn_convolution', ['cudnn_convolution.cpp'])],
#       cmdclass={'build_ext': BuildExtension})


from paddle.utils.cpp_extension import CUDAExtension, setup

# setup(
#     name='cudnn_convolution_custom',
#     ext_modules=CUDAExtension(
#         sources=['cudnn_convolution.cpp']
#     )
# )

setup(
    name='cudnn_convolution_custom',
    ext_modules=CUDAExtension(
        sources=['cudnn_convolution.cpp']
    )
)
