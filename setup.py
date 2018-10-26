#!/usr/bin/env python
from setuptools import setup
import pytransform


if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    setup(name='pytransform',
          version=pytransform.__version__,
          author='Alexander Fabisch',
          author_email='afabisch@googlemail.com',
          url='https://git.hb.dfki.de/besman/pytransform',
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
          packages=['pytransform'],
          requires=["numpy", "scipy", "lxml", "bs4"]
    )
