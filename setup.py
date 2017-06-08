#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.rst').read()
doclink = """
Documentation
-------------

The full documentation is at http://astrofunc.rtfd.org."""
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

setup(
    name='astrofunc',
    version='0.1.0',
    description='This repo contains functions used in astronomy with a special emphasis on lensing profiles.',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='Simon Birrer',
    author_email='sibirrer@gmail.com',
    url='https://github.com/sibirrer/astrofunc',
    packages=[
        'astrofunc', 'astrofunc.LensingProfiles', 'astrofunc.LightProfiles', 'astrofunc.Footprint'
    ],
    package_dir={'astrofunc': 'astrofunc'},
    include_package_data=True,
    install_requires=[
    ],
    license='MIT',
    zip_safe=False,
    keywords='astrofunc',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)
