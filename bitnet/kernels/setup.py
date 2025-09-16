import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Set CUDA architecture if not specified
if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
    os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5'

# Windows-specific compiler flags
if os.name == 'nt':
    extra_compile_args = {
        'cxx': ['/O2', '/MD'],
        'nvcc': [
            '-O3',
            '--use_fast_math',
            '--expt-relaxed-constexpr',
            '-Xcompiler', '/O2',
            '-Xcompiler', '/MD'
        ]
    }
else:
    extra_compile_args = {
        'cxx': ['-O3', '-fopenmp'],
        'nvcc': [
            '-O3',
            '--use_fast_math',
            '--expt-relaxed-constexpr',
            '-Xcompiler', '-O3',
            '-Xcompiler', '-fopenmp'
        ]
    }

setup(
    name='bitnet_kernels',
    ext_modules=[
        CUDAExtension(
            name='bitnet_kernels',
            sources=[
                'bitnet_kernels/bitnet_linear.cu',
                'bitnet_kernels/bitnet_linear_cuda.cpp'
            ],
            include_dirs=['bitnet_kernels'],
            extra_compile_args=extra_compile_args
        ),
        CUDAExtension(
            name='attention_kernels',
            sources=[
                'attention_kernels/attention_scores.cu',
                'attention_kernels/attention_scores_cuda.cpp'
            ],
            include_dirs=['attention_kernels'],
            extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 