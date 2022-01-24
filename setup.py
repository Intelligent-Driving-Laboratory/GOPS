import os

from setuptools import setup, find_packages

import gops


def find_data_packages(where):
    data_packages = []
    for filepath, dirnames, filenames in os.walk(where):
        filepath = filepath.replace('\\', '/') + '/*'
        filepath = filepath[len(where) + 1:]
        data_packages.append(filepath)
    return data_packages


setup(
    name='gops',
    version=gops.__version__,
    description='General Optimal control Problem Solver (GOPS)',
    url='https://gitee.com/tsinghua-university-iDLab-GOPS/gops',
    author='Intelligent Driving Lab (iDLab)',
    packages=[package for package in find_packages() if package.startswith('gops')],
    package_data={
        'gops.env': find_data_packages('gops/env')
    },
    install_requires=[
        'torch>=1.8.0,<=1.10.1',
        'numpy<=1.22.1',
        'ray>=1.0.0,<=1.9.2',
        'gym>=0.17.0,<0.20.0',
        'box2d<=2.4.1',
        'pandas<=1.5.0',
        'tensorboard<=2.8.0',
        'matplotlib<=3.5.1',
        'pyglet<=1.5.21'
    ],
    python_requires='>=3.6',
)
