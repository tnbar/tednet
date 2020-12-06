# -*- coding: UTF-8 -*-

from setuptools import setup, find_packages
from tednet import __author__, __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
      name='tednet',
      version=__version__,
      description='tednet: a framework of tensor decomposition network.',
      author=__author__,
      maintainer=__author__,
      url='https://github.com/perryuu/tednet',
      packages=find_packages(exclude=['docs', "tests"]),
      py_modules=[],
      long_description=long_description,
      long_description_content_type="text/markdown",
      license="MIT",
      platforms=["any"],
      install_requires = ["torch>=1.0.0"]
)
