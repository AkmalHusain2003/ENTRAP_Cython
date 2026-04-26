import numpy as np
from setuptools import setup, find_packages
from Cython.Build import cythonize
from Cython.Compiler import Options

Options.docstrings = False
Options.annotate = False

COMPILER_DIRECTIVES = {
    'language_level': '3',
    'boundscheck': False,
    'wraparound': False,
    'nonecheck': False,
    'cdivision': True,
    'embedsignature': False,
    'initializedcheck': False,
}

PYX_MODULES = [
    'entrap/_core.pyx',
    'entrap/intrinsic_dim.pyx',
    'entrap/tda.pyx',
    'entrap/utils.pyx',
    'entrap/memory.pyx',
    'entrap/tracker.pyx',
    'entrap/dek.pyx',
    'entrap/engine.pyx',
    'entrap/results.pyx',
    'entrap/estimator.pyx',
]

extensions = cythonize(
    PYX_MODULES,
    compiler_directives=COMPILER_DIRECTIVES,
    include_path=[np.get_include()],
    nthreads=0,
)

for ext in extensions:
    ext.include_dirs = [np.get_include()]
    ext.extra_compile_args = ['-O3', '-std=c99', '-fno-math-errno', '-fno-trapping-math']
    ext.extra_link_args = ['-lm']

setup(
    name='entrap',
    version='1.0.0',
    author='Muhammad Akmal Husain',
    license='MIT',
    packages=find_packages(),
    ext_modules=extensions,
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'cython>=3.0.0',
        'scikit-learn>=1.0.0',
        'scipy>=1.7.0',
        'hdbscan>=0.8.27',
        'ripser>=0.6.0',
        'kneed>=0.8.0',
        'joblib>=1.1.0',
        'matplotlib>=3.5.0',
    ],
)
