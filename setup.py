#! /usr/bin/env python

from setuptools import setup

setup(
    name='OSKut',
    packages=['oskut'],
    include_package_data=True,
    version='0.7.0.0',
    install_requires=['tensorflow>=2.0.0', 'pandas',
                      'scipy', 'numpy', 'scikit-learn'],
    license='MIT',
    description='Handling Cross- and Out-of-Domain Samples in Thai Word Segmentation (ACL 2020 Findings) Stacked Ensemble Framework and DeepCut as Baseline model',
    author='Rakpong Kittinaradorn',
    author_email='r.kittinaradorn@gmail.com',
    url='https://github.com/mrpeerat/OSKut',
    keywords=['thai','word segmentation','deep learning'],
    classifiers=[
        'Development Status :: 3 - Alpha'
    ],
)
