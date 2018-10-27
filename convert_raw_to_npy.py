"""
Data loader

License is from https://github.com/aboulch/tec_prediction
"""

import numpy as np
import os
import os.path
import h5py
from PIL import Image

###########################################
###########################################
# Parameters
###########################################
root_dir = "path_to_raw_data"
dest_dir = "path_to_numpy_maps_directory"
imsize = (72, 72)
apply_compensation = True
###########################################
###########################################


def gunzip_some_file(compressed_file,
                     uncompressed_file,
                     delete_file = 1):
    """Unzip a file."""

    if not os.path.isfile(compressed_file):
        if(compressed_file[-1] == 'Z'):
            new_compressed = compressed_file[:-1] + "gz"
            return gunzip_some_file(new_compressed, uncompressed_file,
                                    delete_file)
        raise RuntimeError("No such file '%s' to uncompress" % compressed_file)
    command = "gunzip -dc %s > %s" % (compressed_file, uncompressed_file)
    retcode = os.system(command)
    if(retcode):
        raise RuntimeError("Could not run '%s'" % command)
    if(delete_file):
        os.remove(compressed_file)
    return


def readTEC(filename):
    """ Read TEC files and return latitudes,
        longitudes, times and TEC values. """
    # Opening and reading the IONEX file into memory
    with open(filename, 'rt') as file:
        linestring = file.readlines()

    # creating a new array without the header and only
    # with the TEC maps
    exponent = 0.1  # Default
    for i, line in enumerate(linestring):
        splitted = line.split()

        if splitted[-1] == 'DESCRIPTION' or splitted[-1] == 'COMMENT':
            continue
        if splitted[-1] == 'FILE':
            if splitted[-2] == 'IN':
                NumberOfMaps = int(splitted[0])
                continue
        if splitted[-1] == 'DHGT':
            continue
        if splitted[-1] == 'EXPONENT':
            exponent = pow(10, float(splitted[0]))
            continue
        if splitted[-1] == 'DLAT':
            startLat = float(splitted[0])
            endLat = float(splitted[1])
            stepLat = float(splitted[2])
            continue
        if splitted[-1] == 'DLON':
            startLon = float(splitted[0])
            endLon = float(splitted[1])
            stepLon = float(splitted[2])
            continue
        if splitted[-1] == 'MAP' and (splitted[-4]+splitted[-2] == 'EPOCHFIRST'):
            startYear = float(splitted[0])
            startMonth = float(splitted[1])
            startDay = float(splitted[2])
            date = startYear*366.+startMonth*31.+startDay
            continue
        if splitted[0] == 'END':
            if splitted[2] == 'HEADER':
                break

    NewLongList = linestring[i+1:]
    # Variables that indicate the number of points in Lat. and Lon.
    # 3D array that will contain TEC/RMS values only
    lonarray = np.arange(startLon, endLon+stepLon, stepLon)
    latarray = np.arange(startLat, endLat+stepLat, stepLat)
    pointsLon = lonarray.shape[0]
    pointsLat = latarray.shape[0]
    times = np.zeros(NumberOfMaps, dtype='float32')
    tecdata = np.zeros((NumberOfMaps, pointsLat, pointsLon))
    rmsdata = np.zeros((NumberOfMaps, pointsLat, pointsLon))
    start_fill = False
    for line in NewLongList:
        splitted = line.split()

        if splitted[0] == 'END' and splitted[2] == 'FILE':
            break
        if splitted[-1] == 'MAP' and splitted[-4] == 'START':
            start_fill = True
            # found map start filling
            if splitted[-2] == 'TEC':
                fillarray = tecdata
            else:
                if splitted[-2] == 'RMS':
                    fillarray = rmsdata
                else:
                    start_fill = False
                    # something else
                    continue
            mapnr = int(splitted[0])-1
            continue
        if start_fill:
            if splitted[-1] == 'MAP' and splitted[1] == 'END':
                start_fill = False
                continue
            if splitted[-1] == 'MAP' and splitted[-4] == 'EPOCH':
                times[mapnr] = float(splitted[3])+float(splitted[4])/60.+float(splitted[5])/3600.
                if (float(splitted[0])*366+float(splitted[1])*31+float(splitted[2]))>date: #next day
                    times[mapnr] += 24.
                continue
            if splitted[-1] == 'LAT/LON1/LON2/DLON/H':
                latidx = np.argmin(np.absolute(latarray-float(line[:8])))
                lonidx = 0
                continue
            datalength = len(splitted)
            fillarray[mapnr, latidx, lonidx:lonidx+datalength] = np.array([float(i)*exponent for i in splitted])
            lonidx += datalength

    return (tecdata, rmsdata, lonarray, latarray, times);


def make_bin_data(compensate_earth=True):
    """ Convert TEC IGS files to bin files, smaller and quicker to load """
    for year in range(2003, 2006):
        for day in range(1, 367):
            print("Day {0:01d}, year {1:01d}".format(day, year))
            filename = "igsg{0:03d}0.{1:02d}i".format(day, year % 100)
            filename2 = "codg{0:03d}0.{1:02d}i".format(day, year % 100)
            folder = "/home/ncherrie/TEC_shou/igs.ensg.ign.fr/pub/igs/products/ionosphere/{0:04d}/{1:03d}/".format(year, day)
            # Look for an existing file
            if not (os.path.exists(folder + filename + ".Z")):
                filename = filename2
            if not (os.path.exists(folder + filename + ".Z")):
                filename = "jplg{0:03d}0.{1:02d}i".format(day, year % 100)
            if not (os.path.exists(folder + filename + ".Z")):
                filename = "esag{0:03d}0.{1:02d}i".format(day, year % 100)
            if not (os.path.exists("Original_data/" + filename) or os.path.exists("Original_data/" + filename2)):
                os.system('gzip -c -d -k < ' + folder + filename + ".Z > Original_data/" + filename)
            if not os.path.exists("TEC_data/tecdata_{0:04d}_{1:01d}.bin".format(year,day)):
                try:
                    tecdata, _, _, _, _ = readTEC("Original_data/" + filename)
                except:
                    print("Nothing to do....., day {0:01d} year {1:04d}".format(day, year),filename)
                    continue
                with h5py.File("TEC_data/tecdata_{0:04d}_{1:01d}.bin".format(year, day), 'w') as file:
                    for i in range(12):
                        # Compensate rotation before saving
                        tecdata[i, :, :] = np.roll(tecdata[i, :, :], (int)(73 * i / 12), axis=1)
                    file.create_dataset("tecdata_{0:04d}_{1:01d}".format(year, day), data=tecdata)



if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

fnames = []
for path, subdirs, files in os.walk(root_dir):
    for name in files:

        filename = os.path.join(path, name)

        if (filename[-1] != 'Z') or ("igsg" not in filename):
            continue

        fnames.append(filename)

fnames.sort()

# fnames = ["/data01/tec_data/raw_tec_maps/2015/005/igsg0050.15i.Z"]

count = 0
for filename in fnames:
    print(filename)

    gunzip_some_file(filename, filename[:-2], delete_file=0)

    try:
        data, _, _, _, _ = readTEC(filename[:-2])
        os.remove(filename[:-2])  # clear file
    except:
        print("Nothing to do.....", filename)
        os.remove(filename[:-2])  # clear file
        continue

    if apply_compensation:
        for i in range(data.shape[0]):
            data[i] = np.roll(data[i], (int)(73 * i / 12 + 73/2), axis=1)

    output = np.zeros((data.shape[0],)+imsize)

    if imsize is not None:
        data *= 100  # milimeters
        data = data.astype(np.uint16)
        for i in range(data.shape[0]):
            im = Image.fromarray(data[i])
            im = im.convert(mode='I')
            im = im.resize(imsize)
            output[i] = np.array(im, dtype=np.float32)/100  # in meters

    fname = filename.split("/")
    year = fname[-3]
    day = fname[-2]
    fname = os.path.join(dest_dir, "tecdata_{}_{}".format(year, day))
    np.save(fname, output)
