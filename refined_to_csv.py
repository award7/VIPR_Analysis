#!/usr/bin/env python
# coding: utf-8

import pandas as pd

def refined_to_csv(df_raw, input_file, final_voxel_list):
    #write refined df to new csv file
    
    df_refined = df_raw[df_raw['Voxel'].isin(final_voxel_list)]
    
    rename_file = input_file.replace("_raw.csv", "_refined.csv")
    df_refined.to_csv(rename_file, sep = ",", index = False)
        
    return(df_refined)