#!/usr/bin/env python


from distutils.core import setup
import pytransform


if __name__ == "__main__":
    setup(name='pytransform',
          version=pytransform.__version__,
          author='Alexander Fabisch',
          author_email='afabisch@googlemail.com',
          url='https://git.hb.dfki.de/besman/pytransform',
          description='3D transformations for Python',
          long_description=open('README.md').read(),
          license='New BSD',
          packages=['pytransform'],
          requires=["numpy", "scipy", "lxml", "bs4"]
    )
