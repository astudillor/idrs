#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="idrs",
    version="1.0.0",
    author="Reinaldo Astudillo",
    author_email="R.A.Astudillo@tudelft.nl",
    description="IDR method for solving systems of linear equations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/astudillor/idrs",
    install_requires=['numpy (>=1.11)', 'scipy (>=0.17)'],
    packages=setuptools.find_packages(),
    zip_safe=False
)
