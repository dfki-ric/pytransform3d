#!/usr/bin/env python


from distutils.core import setup
import pytransform


if __name__ == "__main__":
    setup(name='pytransform',
          version=pytransform.__version__,
          author='Alexander Fabisch',
          author_email='afabisch@googlemail.com',
          url='https://git.hb.dfki.de/besman/pytransform',
          description='Conversions between various transformation '
                      'representations',
          long_description=open('README.rst').read(),
          license='New BSD',
          packages=['pytransform'],)
