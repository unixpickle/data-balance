"""
Package meta-data.
"""

from setuptools import setup

setup(
    name='data-balance',
    version='0.0.1',
    description='Experiments with data balancing.',
    url='https://github.com/unixpickle/data-balance',
    author='Alex Nichol',
    author_email='unixpickle@gmail.com',
    license='MIT',
    packages=['data_balance'],
    install_requires=['Pillow', 'numpy', 'sklearn']
)
