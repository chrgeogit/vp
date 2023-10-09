# This is a sample Python script.
import math
import os

import geopandas as geopandas
import matplotlib.pyplot as plt
import np as np
import numpy as np
from scipy.interpolate import griddata
# import packages
import pandas as pd
from geopy import Point
from haversine import haversine
from geopy.distance import geodesic as GD, geodesic
import numpy as np
from datetime import datetime

plt.style.use("seaborn-v0_8-whitegrid")


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def distance(s_lat, s_lng, e_lat, e_lng):
    # approximate radius of earth in km
    R = 6373.0

    s_lat = s_lat * np.pi / 180.0
    s_lng = np.deg2rad(s_lng)
    e_lat = np.deg2rad(e_lat)
    e_lng = np.deg2rad(e_lng)

    d = np.sin((e_lat - s_lat) / 2) ** 2 + np.cos(s_lat) * np.cos(e_lat) * np.sin((e_lng - s_lng) / 2) ** 2

    return 2 * R * np.arcsin(np.sqrt(d))


def get_angle(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # find path
    path, dirs, files = next(os.walk("./data/"))
    file_count = len(files)
    # create empty list
    dataframes_list = []
    # append datasets to the list
    for i in range(file_count):
        temp_df = pd.read_csv("./data/" + files[i])
        dataframes_list.append(temp_df)
    # dataframes
    ais = pd.read_csv("./data/ais_points_masked.csv")
    copernicus = pd.read_csv("./data/copernicus_weather_conv.csv")
    wni = pd.read_csv("./data/wni_weather_conv.csv")
    crewreport = pd.read_csv("./data/reported_masked.csv")

    # Filter Rows using DataFrame.query() - get single ship dataset
    ais1 = ais[(ais.imo == 9000101)]
    ais1.to_csv('ais1.csv')

    ais2 = ais[(ais.imo == 9000102)]
    ais2.to_csv('ais2.csv')

    ais3 = ais[(ais.imo == 9000103)]
    ais3.to_csv('ais3.csv')

    ais4 = ais[(ais.imo == 9000104)]
    ais4.to_csv('ais4.csv')

    ais5 = ais[(ais.imo == 9000105)]
    ais5.to_csv('ais5.csv')

    ais6 = ais[(ais.imo == 9000106)]
    ais6.to_csv('ais6.csv')

    ais7 = ais[(ais.imo == 9000107)]
    ais7.to_csv('ais7.csv')

    ais8 = ais[(ais.imo == 9000108)]
    ais8.to_csv('ais8.csv')

    p = ais.dtypes
    ais8.head()
    BBox = (ais8.lon.min(), ais8.lon.max(), ais8.lat.min(), ais8.lat.max())
    # Plotting the points on the graph
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(ais8.lon, ais8.lat, 'xb-')

    # Setting limits for the plot
    ax.set_xlim(BBox[0], BBox[1])
    ax.set_ylim(BBox[2], BBox[3])

    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid()
    # Data for ship 9000108
    tline = ais8["time"]
    xline = ais8["lon"]
    yline = ais8["lat"]
    # ax.plot3D(xline, yline, tline, 'gray')

    # Data for three-dimensional scattered points
    # ais8["data"] = pd.to_datetime(ais8["date"])
    # format_str = "%Y-%m-%dT%H:%M:%S"
    # zdata = [datetime.strptime(d, format_str) for d in ais8["time"]]
    # df['Distance'] = df.apply(lambda row: haversine(row['Latitude'], row['Longitude'], 40.7128, -74.0060), axis=1)
    # df["distance"] = df[["geo1", "geo2"]].apply(lambda x: haversine(*x.geo1, *x.geo2), axis="columns")

    # Filter Rows using  - get single ship reported_masked

    crewreport8 = crewreport[(crewreport.imo == 9000108)]

    # convert time data, element = datetime.datetime.strptime(string,"%d/%m/%Y")
    # 2023-01-01T00:11:45
    crewreport8['timestamp1'] = pd.to_datetime(crewreport8['report_start'], format='%Y-%m-%dT%H:%M:%S')
    crewreport8.to_csv('crewreport8.csv')
    time_marks8 = crewreport8['timestamp1']
    time_marks8str = time_marks8.dt.strftime("%Y-%m-%d %H:%M:%S")

    ####################################################
    # shift time lat and lon column one row upwards
    ais8['shiftlat'] = ais8["lat"].shift(-1)
    ais8['shiftlon'] = ais8["lon"].shift(-1)
    ais8['timestamp'] = pd.to_datetime(ais8['time'], format='%Y-%m-%dT%H:%M:%S')

    ais8['shifttimestamp'] = ais8["timestamp"].shift(-1)
    # ais8['shifttimestamp'] = pd.to_datetime(ais8['time'], format='%Y-%m-%dT%H:%M:%S')

    # get rid of nan values due to shifting
    ais8 = ais8.dropna()
    # create new column : geodesic distance between succesive points
    ais8['distance_miles'] = ais8[['lat', 'lon', 'shiftlat', 'shiftlon']].apply(
        lambda x: geodesic((x[0], x[1]), (x[2], x[3])).miles, axis=1)
    # create new column for delta time , between two succesive points
    ais8['dt'] = ais8['shifttimestamp'] - ais8['timestamp']
    # covert dt to float, hours
    ais8['dtfloat'] = ais8['dt'].dt.total_seconds() / 3600
    # new column for valocity calculation
    ais8['dv'] = ais8['distance_miles'] / ais8['dtfloat']
    # velocity vector angle
    ais8['angle'] = ais8[['lat', 'lon', 'shiftlat', 'shiftlon']].apply(
        lambda x: get_angle(x[0], x[1], x[2], x[3]), axis=1)
    #    ais8['shiftpoints'] = ais8['points'].shift(1)

    ais8.to_csv('ais8shift.csv')
    ais8slice = ais8[['shiftlat', 'shiftlon', 'timestamp', 'shifttimestamp', 'distance_miles']]
    ais8slice.to_csv('ais8slice.csv')
    # convert time data, element = datetime.datetime.strptime(string,"%d/%m/%Y")
    # 2023-01-01T00:11:45
    # ais8shift['timestamp'] = pd.to_datetime(ais8['time'], format='%Y-%m-%dT%H:%M:%S')
    ais8xytslice = ais8slice[['shiftlat', 'shiftlon', 'shifttimestamp']]
    # ais8xytslice = pd.DataFrame(ais8xytslice, columns=['timestamp', 'lat', 'lon'])
    ais8xytslice['shifttimestamp'] = pd.to_datetime(ais8xytslice['shifttimestamp'])
    ais8xytslice.set_index('shifttimestamp', inplace=True)
    # ais8xytinterpol = ais8xytslice.reindex(time_marks8str)
    interpolationmarks = pd.to_datetime(time_marks8)
    interpolated_values8 = ais8xytslice.resample(time_marks8).interpolate()
    interpolated_values8.reset_index(inplace=True)
    # interpolated_lat = np.interp(
    #     time_marks8,
    #     ais8xytslice['shifttimestamp'].values,
    #     ais8xytslice['shiftlat'].values,
    #     left= ais8xytslice['shiftlat'].values[0],
    #     right=ais8xytslice['shiftlat'].values[-1],
    # )
    #
    # interpolated_lon = np.interp(
    #     time_marks8,
    #     ais8xytslice['shifttimestamp'].values,
    #     ais8xytslice['shiftlon'].values,
    #     left= ais8xytslice['shiftlon'].values[0],
    #     right=ais8xytslice['shiftlon'].values[-1],
    # )
    new_rows = {
        'shifttimestamp': time_marks8,
        'shiftlat': interpolated_lat,
        'shiftlon': interpolated_lon
    }
    interpolated_values8 = pd.DataFrame(new_rows)



    def interpol_values(timestamp):
        matchrows =  ais8xytslice.loc[ais8xytslice['shifttimestamp'] == timestamp]
        return matchrows.interpolate(method='time',limit_direction='both')

    interpolated_values8 = [interpol_values(timestamp) for timestamp in interpolationmarks]
    ais8xytinterpol = pd.concat(interpolated_values8, ignore_index=True)

    for timemark in time_marks8:
        interpolrow = ais8xytslice.loc[ais8xytslice['shifttimestamp']] == timemark.copy()
        if not interpolrow.empty:
            ais8xytinterpol = pd.concat([ais8xytinterpol, interpolrow])

        # interpolated_values8.append(interpolrow)

    #     interpolated_values8.append((timemark, interlat, interlon))
    ais8xytinterpol = ais8xytslice.interpolate(method='time', limit_direction='both', time=time_marks8)
    ais8xytinterpol.reset_index(inplace=True)
    ais8xytinterpol.to_csv('ais8xytinterpol.csv')
    # use wni dataset
    # convert minutes to decimal degrees
    wni['lat_deg'] = wni['lat_min'] / 60
    wni['lon_deg'] = wni['lon_min'] / 60
    # save column to existing dataset
    wni.to_csv('wni_with_deg.csv')

    #####

    ais8resample = ais8slice
    #
    zdata = pd.to_numeric(pd.to_datetime(ais8["time"]))
    zdata32 = zdata[0:1000]
    # zdata = pd.to_datetime(ais8["time"])
    xdata = ais8["lon"][0:1000]
    ydata = ais8["lat"][0:1000]
    # ax.scatter(xdata, ydata, zdata32)
    # plt.show()
    print(ais8.dtypes)

    # display datasets
    # for dataset in dataframes_list:
    #     print(dataset)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
