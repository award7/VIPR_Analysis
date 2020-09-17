#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from scipy import stats

def rm_outliers(df):
    threshold = 3
    df = df[(np.abs(stats.zscore(df)) < threshold).all(axis = 1)]
    df_rm_outliers = df[(df > 0).all(1)]
	
    return(df_rm_outliers)
