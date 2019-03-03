# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: io.py

@time: 2019/2/9 13:39

@desc:

"""

from os import path
import json
import codecs
import numpy as np
import pickle


def format_filename(_dir, filename_template, **kwargs):
    """Obtain the filename of data base on the provided template and parameters"""
    filename = path.join(_dir, filename_template.format(**kwargs))
    return filename


def pickle_load(filename):
    try:
        with open(str(filename), 'rb') as f:
            obj = pickle.load(f)

        print('Logging Info - Loaded:', filename)

    except EOFError:
        print('Logging Warning - Cannot load:', filename)
        obj = None

    return obj


def pickle_dump(filename, obj):
    with open(str(filename), 'wb') as f:
        pickle.dump(obj, f)

    print('Logging Info - Saved:', filename)


def write_log(filename, log, mode='w'):
    with codecs.open(filename, mode, encoding='utf8') as writer:
        writer.write('\n')
        json.dump(log, writer, indent=4, default=str, ensure_ascii=False)
    print('Logging Info - log saved in', filename)
