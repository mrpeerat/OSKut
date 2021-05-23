#! /usr/bin/env python

from setuptools import setup

with open('README.md', 'r', encoding='utf-8-sig') as f:
    readme = f.read()

setup(
    name='OSKut',
    packages=['oskut'],
    include_package_data=True,
    version='1.0',
    long_description=readme,
    long_description_content_type='text/markdown',
    install_requires=[
        'tensorflow>=2.0.0',
        'pandas',
        'scipy',
        'numpy',
        'scikit-learn',
        'pyahocorasick<=1.4.0'
    ],
    license='MIT',
    package_data={
        'oskut': [
            'model/*',
            'variable/*',
            'weight/*',
            'deepcut/weight/*'
        ],
    },
    description='Handling Cross- and Out-of-Domain Samples in Thai Word Segmentation (ACL 2020 Findings) Stacked Ensemble Framework and DeepCut as Baseline model',
    author='Peerat Limkonchotiwat',
    author_email='peerat.limkonchotiwat@gmail.com',
    url='https://github.com/mrpeerat/OSKut',
    keywords=['thai','word segmentation','deep learning'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3',
        'Natural Language :: Thai',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic'
    ],
)
