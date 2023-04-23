from distutils.core import setup
from setuptools import find_packages

import os

cwd = os.getcwd()

setup(
    name='Diffusion236610',
    version='1.0',
    description=(
        'Code package for the final project in a diffusion-based learning course (236610).'
    ),
    author='Yonatan Elul',
    author_email='renedal@gmail.com',
    url='https://github.com/YonatanE8/236610.git',
    license='',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Private',
        'Topic :: Software Development :: Dynamical Systems Modelling',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    package_dir={'Diffusion236610': os.path.join(cwd, 'Diffusion236610')},
    packages=find_packages(
        exclude=[
            'data',
            'logs',
        ]
    ),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'torch',
        'torchvision',
        'tensorboard',
        'tqdm',
        'h5py',
        'pandas',
    ],
)
