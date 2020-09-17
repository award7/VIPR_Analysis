#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

def sumproduct(df, df_hr):
    row_count = len(df.index)
    cols = ['Flow']
    df_flow = pd.DataFrame(columns = cols, index = range(row_count))
    
    #calculate hr
    hr = 60 / df_hr.iat[0,20]
    
    #setup absolute hr vectors
    v1 = np.array(df_hr.iloc[0][2:21].astype(float))
    v2 = np.array(df_hr.iloc[0][1:20].astype(float))
    
    for row in range(0, row_count):
        v3 = np.array(df.iloc[row][2:21].astype(float))
        v4 = np.array(df.iloc[row][1:20].astype(float))
        v5 = v1 - v2
        v6 = v3 + v4
        v7 = v5 * v6
        flow_beat = sum(v7)*0.5
        flow = flow_beat * hr
        df_flow.loc[row].Flow = flow
    
    return(df_flow)

