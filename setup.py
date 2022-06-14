#!/usr/bin/env python
from setuptools import setup, find_packages
import pytransform3d


if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    setup(name="pytransform3d",
          version=pytransform3d.__version__,
          author='Alexander Fabisch',
          author_email='afabisch@googlemail.com',
          url='https://github.com/dfki-ric/pytransform3d',
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
          license='BSD-3-Clause',
          packages=find_packages(),
          install_requires=["numpy", "scipy", "matplotlib", "lxml",
                            "beautifulsoup4"],
          extras_require={
              "all": ["pydot", "trimesh", "open3d"],
              "doc": ["numpydoc", "sphinx", "sphinx-gallery", "sphinx-bootstrap-theme"],
              "test": ["nose", "coverage"]
          }
    )
