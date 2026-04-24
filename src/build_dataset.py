import csv
import os
from utils import readcsv, savecsv


if __name__ == "__main__":
    filename_list = ['stealthyIMU_air.csv', 'stealthyIMU_navigation_ny.csv', 'stealthyIMU_navigation_sd.csv', 'stealthyIMU_reminder.csv', \
    'stealthyIMU_stock.csv', 'stealthyIMU_sun.csv', 'stealthyIMU_time.csv', 'stealthyIMU_weather.csv']

    tot = 0
    data_list = list()
    for filename in filename_list:
        filename = os.path.join("./metadata", filename)
        data_list_now = readcsv(filename)
        print(data_list_now[0])
        print(len(data_list_now))
        tot += len(data_list_now)
        data_list_new = list()
        for i in range(len(data_list_now)):
            filepath_now = data_list_now[i][2].split('/data/kesun_StealthyIMU/')
            filepath_now = os.path.join("./data", filepath_now[1])
            print(filepath_now)
            data_list_now[i][2] = filepath_now
        data_list = data_list + data_list_now

    print(len(data_list))
    print(tot)
    savecsv(os.path.join("./metadata", "stealthyIMU_all_relative.csv"), data_list)