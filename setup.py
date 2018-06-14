#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools

short_description = "IDR(s) method for solving systems of linear equations"
minimun_requirements = ['numpy>=1.11', 'scipy>=0.17']


def get_data(filename, default_data, action_data=lambda x: x):
    data = ""
    try:
        with open(filename, "r") as fhandle:
            data = action_data(fhandle.read())
    except IOError:
        data = default_data
    return data


def get_long_description():
    return get_data(filename='README.md',
                    default_data=short_description)


def get_requirements():
    return get_data(filename='requirements.txt',
                    default_data=minimun_requirements,
                    action_data=lambda x: x.strip().split('\n'))


setuptools.setup(
    name="idrs",
    version="1.0.0",
    author="Reinaldo Astudillo",
    author_email="R.A.Astudillo@tudelft.nl",
    description=short_description,
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/astudillor/idrs",
    install_requires=get_requirements(),
    packages=setuptools.find_packages(),
)
