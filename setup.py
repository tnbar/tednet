# -*- coding: UTF-8 -*-

from setuptools import setup, find_packages


__author__ = "Perry"
__version__ = "0.0.1"

setup(
      name='tednet',
      version=__version__,
      description='tednet: a framework of tensor decomposition network.',
      author=__author__,
      maintainer=__author__,
      url='https://github.com/perryuu/tednet',
      packages=find_packages(),
      py_modules=[],
      long_description="A toolkit of tensor decomposition network.",
      license="MIT",
      platforms=["any"],
      install_requires = ["torch>=1.0.0"]
)