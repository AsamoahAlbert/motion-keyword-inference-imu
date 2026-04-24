"""
Utility functions for dataset loading and saving.

Includes:
- readcsv
- savecsv
"""

import csv
import os

def savecsv(filename, savedata):
    fwrite_data = open(filename, 'w', newline='')
    writecsv_data = csv.writer(fwrite_data)
    # writecsv_data.writerow(fields)
    for i in range(len(savedata)):
        writecsv_data.writerow(savedata[i])
    fwrite_data.close()

def readcsv(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        data_list = list(reader)
    # label = data_list[0]
    return data_list
