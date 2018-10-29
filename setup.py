#!/usr/bin/env python
from setuptools import setup
import pytransform3d


if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    setup(name='pytransform3d',
          version=pytransform3d.__version__,
          author='Alexander Fabisch',
          author_email='afabisch@googlemail.com',
          url='https://github.com/rock-learning/pytransform3d',
          description='3D transformations for Python',
          long_description=long_description,
          long_description_content_type="text/markdown",
          classifiers=[
              "Programming Language :: Python :: 2",
              "Programming Language :: Python :: 3",
              "License :: OSI Approved :: BSD License",
              "Operating System :: OS Independent",
              "Topic :: Scientific/Engineering :: Mathematics",
              "Topic :: Scientific/Engineering :: Visualization",
          ],
          license='New BSD',
          packages=['pytransform3d'],
          requires=["numpy", "scipy", "lxml", "bs4"]
          )
