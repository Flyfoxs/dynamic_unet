#!/usr/bin/env python
from setuptools import setup
import setuptools

try:
    with open('README.md') as file:
        long_description = file.read()
except Exception as e:
    print('Readme read failed')

setup(
    name='dynamic_unet',
    version='0.1.2',
    description='dynamic_unet',
    long_description='dynamic_unet: https://github.com/Flyfoxs/dynamic_unet',
    url='https://github.com/Flyfoxs/dynamic_unet',
    author='Felix Li',
    author_email='lilao@163.com',
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    install_requires=[
        "termcolor>=1.1",
        "Pillow==10.0.1",  # torchvision currently does not work with Pillow 7
        "yacs>=0.1.6",
        "tabulate",
        "easydict",
        "nibabel",
        "pydicom",
        "cloudpickle",
        "matplotlib",
        "tqdm>4.29.0",
        "tensorboard",
        "fvcore",
        "future",  # used by caffe2
        "pydot",  # used to save caffe2 SVGs
        "SimpleITK",
        "plotly",
    ],
    keywords='unet fastai',
    packages=setuptools.find_packages(),
)