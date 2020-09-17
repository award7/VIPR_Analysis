#!/usr/bin/env python
# coding: utf-8

import os
import errno

def silent_remove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise