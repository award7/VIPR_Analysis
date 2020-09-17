#!/usr/bin/env python
# coding: utf-8

import csv
import os

def split_workbook(wb, fpath):

    #split summary.xls tabs into individual .csv files
    for i in range(3, wb.nsheets):
        sheet = wb.sheet_by_index(i)
        fname = "%s_raw.csv" %(sheet.name.replace(" ","_"))
        with open(os.path.join(fpath, fname), "w", newline = '') as file:
            writer = csv.writer(file, delimiter = ",")
            header = [cell.value for cell in sheet.row(0)]
            writer.writerow(header)
            for row_idx in range(1, sheet.nrows):
                row = [cell.value for cell in sheet.row(row_idx)]
                writer.writerow(row)