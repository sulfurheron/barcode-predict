from setuptools import setup
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


setup(name='barcode-predict',
      packages=[],
      install_requires=[],
      description='Repository for training barcode recognition models',
      author='Kindred AI',
      url='https://github.com/sulfurheron/barcode-predict',
      author_email='',
      version='0.0.1')
