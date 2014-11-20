#!/usr/bin/env python


from distutils.core import setup
import pytransform


if __name__ == "__main__":
    setup(name='pytransform',
          version=pytransform.__version__,
          author='Alexander Fabisch',
          author_email='afabisch@googlemail.com',
          url='TODO',
          description='TODO',
          long_description=open('README.rst').read(),
          license='New BSD',
          packages=['pytransform'],)
