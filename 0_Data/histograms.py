# To execute: python3 histograms.py -k keyfile.key

from constants import HIST_BINS_LIST, NUM_IMGS_PER_YEAR, NUM_TEMP_BANDS, NUM_REF_BANDS, CROP_SENTINEL, GBUCKET, RED_BAND, NIR_BAND
import numpy as np
from osgeo import gdal # I need to run it with python3. 
import matplotlib
matplotlib.use('Agg')
import sys
import matplotlib.pyplot as plt
from os import listdir, mkdir
from os.path import isfile, join, getsize, expanduser, basename, isdir
import argparse
import datetime


NUM_YEARS_EXTEND_FORWARD = 0
SUMMARY_STAT_NAMES = ['max', 'mean', 'min']

def day_of_year_to_yyyy_mm_dd(dates, start_year=2003):
    year = start_year
    previous_day = 0
    result = []

    for day_of_year in dates:
        # Check if we need to move to the next year
        if day_of_year < previous_day:
            year += 1
        previous_day = day_of_year

        # Using the calculated year and day within the year
        date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(day_of_year) - 1)
        result.append(date.strftime('%Y-%m-%d'))
    
    return result

def visualize_histogram(hist, title, save_folder, show=False):
    """
    Outputs an image of a histogram's multiple bands for a sanity check
    """
    num_bands = hist.shape[2]
    f, axarr = plt.subplots(num_bands, sharex=True)
    for band in range(num_bands):
        axarr[band].imshow(hist[:,:,band])
    plt.suptitle(title)
    plt.savefig(join(save_folder, title +'.png'))
    if show:
        plt.show()

def return_files(path):
    return [f for f in listdir(path) if (isfile(join(path, f)) and not f.startswith('.'))]


def analyze_histograms(directory, save_directory):
    hist_files = return_files(directory)
    count = 0
    for idx, f in enumerate(hist_files):
        histograms = np.load(join(directory, f))
        shape = histograms.shape
        if histograms.ndim < 2:
            print(f"Skipping {f} due to unexpected dimensions: {histograms.shape}")
            continue  # Skip this file and move to the next
        if idx == 0:
            num_bins = shape[0]
            num_bands = shape[2]
            histogram_sums = np.zeros(( num_bins, num_bands))
            mode_matrix = np.empty((num_bands,  len(SUMMARY_STAT_NAMES), len(hist_files)))
        histogram_sums += np.sum(histograms, axis=1) #sum along time axis

        place = basename(f)[:-4]
        modes = np.argmax(histograms, axis=0)
        maxes = np.max(modes, axis=0)
        means = np.average(modes, axis=0)
        mins = np.min(modes, axis=0)
        mode_matrix[:, 0, idx] = maxes
        mode_matrix[:, 1, idx] = means
        mode_matrix[:, 2, idx] = mins
        visualize_histogram(histograms, place, save_directory)
        plt.clf()
        
        count += 1
        print(place, shape, str(count) + '/' + str(len(hist_files)))
        print(' '.join('{:02}'.format(int(m)) for m in maxes), 'max mode bins')
        print(' '.join('{:02}'.format(int(m)) for m in means), 'mean mode bins')
        print(' '.join('{:02}'.format(int(m)) for m in mins), 'min mode bins')
        print()
    
    #plot summed histograms
    plt.figure()
    for band in range(num_bands):
        plt.subplot(num_bands, 1, band+1)
        plt.bar(list(range(len(histogram_sums[:, band]))), histogram_sums[:, band])
        plt.ylabel(str(band))
        plt.yticks([])
        plt.xticks(list(range(0, num_bins, 2)))
    plt.suptitle('Histogram density sums')
    plt.savefig(join(save_directory, '111_density_sums.png'))

    for band in range(num_bands):
        #plot mode matrix summary stats
        plt.figure()
        for idx, stat_name in enumerate(SUMMARY_STAT_NAMES):
            plt.subplot(len(SUMMARY_STAT_NAMES), 1, idx + 1)
            plt.hist(mode_matrix[band, idx], bins=num_bins)
            plt.ylabel(stat_name)
            plt.xticks(range(0, num_bins, 2))
        save_name = '111_band_' + str(band) + '_mode_visualization.png'
        plt.suptitle('Band ' + str(band) + ' mode histograms')
        plt.savefig(join(save_directory, save_name))
        plt.clf()

def read_key_file(key_file_path):
    params = {}
    with open(key_file_path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):  # Ignore comments and empty lines
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    key, value = parts
                    params[key] = value.split("#")[0].strip()  # This also removes comments from the value, if any
    return params

def return_tif_filenames(path):
    return [f for f in listdir(path) if (isfile(join(path, f)) and not f.startswith('.') and f.endswith('.tif'))]


def read_tif(tif_path):
    """
    Reads tif image into a tensor and attempts to print date information.
    """
    try:
        gdal_dataset = gdal.Open(tif_path)
    except:
        print('Error opening', tif_path, 'with gdal')
        return None
    if gdal_dataset is None:
        print('gdal returned None from', tif_path)
        return None

    # Read image data
    gdal_result = gdal_dataset.ReadAsArray().astype(np.uint16)
    if len(gdal_result.shape) == 2:
        gdal_result = np.reshape(gdal_result, [1] + list(gdal_result.shape))
    image_data = np.transpose(gdal_result, [1, 2, 0])

    # Attempt to extract date information from metadata
    # metadata = gdal_dataset.GetMetadata()
    # if metadata:
    #     print("Metadata keys:", metadata.keys())  

    return image_data  


def calc_histograms(image, bin_seq_list):
    """
    Makes a 3D tensor of pixel histograms [normalized bin values, time, band]
    input is a 3D image in tensor form [H, W, time/num_bands + band]
    """
    num_bands = len(bin_seq_list)
    num_bins = len(bin_seq_list[0]) - 1
    if image.shape[2] % num_bands != 0:
        raise Exception('Number of bands does not match image depth.')
    num_times = image.shape[2] // num_bands
    hist = np.zeros([num_bins, num_times, num_bands])
    for i in range(image.shape[2]):
        band = i % num_bands
        density, _ = np.histogram(image[:, :, i], bin_seq_list[band], density=False)
        total = density.sum()  # normalize over only values in bins
        hist[:, i // num_bands, band] = density / float(total) if total > 0 else 0
    return hist


def mask_image(img, mask, num_bands, num_years_extend_backward, num_years_extend_forward,num_imgs_per_year):
    """
    Masks away non-crop pixels in all 2D slices of 3D image tensor of shape X x Y x (bands/time)
    """
    num_imgs = img.shape[2]//num_bands
    assert num_imgs == int(num_imgs)
    remainder_imgs = num_imgs % num_imgs_per_year
    for t in range(num_imgs):
        mask_year = int((t-remainder_imgs)/num_imgs_per_year)
        if mask_year < num_years_extend_backward:
            mask_slice = mask[:,:,0]
        elif mask_year >= mask.shape[2] + num_years_extend_backward:
            assert mask_year < mask.shape[2] + num_years_extend_backward + num_years_extend_forward
            mask_slice = mask[:,:,-1]
        else:
            mask_slice = mask[:, :, mask_year - num_years_extend_backward] 
        for b in range(num_bands):
            img[:, :, t*num_bands + b] = np.multiply(img[:, :, t*num_bands + b], mask_slice)
    return img

def generate_dates_for_year(year):
    dates = []
    start_date = datetime.datetime(year, 1, 1)
    interval = 365 / 46  # Interval in days between each date
    for i in range(46):
        date = start_date + datetime.timedelta(days=i * interval)
        dates.append(date.strftime('%Y-%m-%d'))
    return dates

def get_places(filenames):
    """
    Gets places of tif imagery from filenames in list. Assumes names are of form
    <country>_<img type>_<place name>.tif|_<date info>.tif
    """
    places = []
    for f in filenames:
        place = f.split('_')[2]
        if place.find('tif') != -1: place = place[:-1]
        places.append(place)
    return places

def collect_tif_path_dict(sat_dir, temp_dir, mask_dir, verbose=True):
    """
    Returns a dictionary of form {name of place : (sat path, temp path, mask path)}
    """
    all_sat_files = return_tif_filenames(sat_dir)
    sat_places = get_places(all_sat_files)
    all_temp_files = return_tif_filenames(temp_dir)
    temp_places = get_places(all_temp_files)
    all_mask_files = return_tif_filenames(mask_dir)
    mask_places = get_places(all_mask_files)
    tif_dict = {}
    for s_i, place in enumerate(sat_places):
        if place not in temp_places or place not in mask_places:
            if verbose: print(place, 'missing temp and/or mask file')
            continue
        t_i = temp_places.index(place)
        m_i = mask_places.index(place)
        tif_dict[place] = (all_sat_files[s_i], all_temp_files[t_i], all_mask_files[m_i])
    return tif_dict


if __name__ == '__main__':
    gdal.SetCacheMax(2**35)

    parser = argparse.ArgumentParser(description="Pull MODIS data for specified countries and imagery types.")
    parser.add_argument("-k", "--key_file", type=str, help="Path to the key file with input parameters.")
    args = parser.parse_args()

    if args.key_file:
            params = read_key_file(args.key_file)
            target_folder_name = params.get('HISTOGRAM_FOLDER')
            modis_data = params.get('DOWNLOAD_FOLDER')
            composite_period = int(params.get('COMPOSITE_PERIOD'))
            use_gee_shapefile = params.get('USE_GEE_SHAPEFILE') == '1'
            region_name = params.get('REGION')

    else:
            print("Key file is required.")
            sys.exit(1)

    print('Creating histograms')
   
    indices_only = False  # Default value.  
    sat_directory = join(modis_data + '/sat/')
    temp_directory = join(modis_data + '/temp/')
    mask_directory = join(modis_data + '/cover/')
    if not isdir(target_folder_name): mkdir(target_folder_name)

    
    tif_dict = collect_tif_path_dict(sat_directory, temp_directory, mask_directory)

    count = 0
    num_tifs = len(tif_dict)
    for place, tif_path_tuple in tif_dict.items():
        hist_save_path = join(target_folder_name, place + '_histogram')
        ndvi_save_path = join(target_folder_name, place + '_ndvi')

        if isfile(hist_save_path + '.npy'): 
            print(place, 'already processed. Continuing...')
            count += 1
            continue 
        
        sat_path, temp_path, mask_path = tif_path_tuple
        if "2003-01-01" not in sat_path and not use_gee_shapefile:
            print('Data must start from 2023-01-01.')
            sys.exit(1)

        sat_tensor = read_tif(sat_directory + sat_path) 
        # Let's extract the dates from the data. 
        indices_to_remove = list(range(7, sat_tensor.shape[2], 8))
        mask_dates = np.ones(sat_tensor.shape[2], dtype=bool)
        mask_dates[indices_to_remove] = False
        dates_raw = sat_tensor[int(sat_tensor.shape[0]/2),int(sat_tensor.shape[0]/2),indices_to_remove]
        if 'mali' in sat_path: dates_raw[604] = 56   # Manual correction because of MODIS error in Mali. Only valid if start_date = 2003-01-01. 
        dates = day_of_year_to_yyyy_mm_dd(np.array(dates_raw,dtype=np.int64))
        # for i in range(len(dates)):
        #     print(i,dates_raw[i],dates[i])
        # years_check = range(2003, 2021)
        # print([f"{year}: {sum(date.startswith(str(year)) for date in dates)}" for year in years_check])
        # exit()

        sat_tensor = sat_tensor[:, :, mask_dates]

        print("tensor_sizes")
        print("Sat: ",np.shape(sat_tensor))
        temp_tensor = read_tif(temp_directory + temp_path)
        print("Temp: ",np.shape(temp_tensor))
        mask_tensor = read_tif(mask_directory + mask_path)
        print("Mask: ",np.shape(mask_tensor))
        if sat_tensor is None or temp_tensor is None or mask_tensor is None: 
            print(place, 'Problem with data. Continuing')
            num_tifs -= 1
            continue 

        if use_gee_shapefile and region_name == 'Argentina':
            # Correct the tensor to cover from 2003-01-01 to 2016-12-31.
            sat_tensor = sat_tensor[:,:,-4508:]
            temp_tensor = temp_tensor[:,:,-1288:]
            mask_tensor = np.tile(mask_tensor, (1, 1, 4))[:,:,:14]
            # Generate dates for each year from 2003 to 2016
            all_dates = []
            for year in range(2003, 2017):
                all_dates.extend(generate_dates_for_year(year))
            dates=all_dates
        
        assert sat_tensor is not None
        assert temp_tensor is not None  
        assert mask_tensor is not None

        if sat_tensor.shape[:2] != temp_tensor.shape[:2] or\
           sat_tensor.shape[:2] != mask_tensor.shape[:2]:
               print(place, 'slice shapes do not match! sat, temp, mask shapes:', sat_tensor.shape[:2], temp_tensor.shape[:2], mask_tensor.shape[:2])
               count += 1
               continue

        mask_tensor[mask_tensor != CROP_SENTINEL] = 0
        mask_tensor[mask_tensor == CROP_SENTINEL] = 1


        num_sat_imgs_orig = sat_tensor.shape[2] // NUM_REF_BANDS
        num_temp_imgs = temp_tensor.shape[2] // NUM_TEMP_BANDS
        # print('Number of images')
        # print(num_sat_imgs_orig)
        # print(num_temp_imgs)
        assert num_sat_imgs_orig == num_sat_imgs_orig 
        num_imgs_per_year = NUM_IMGS_PER_YEAR * 8 / composite_period
        mask_years_missing = int(num_temp_imgs/num_imgs_per_year) - NUM_YEARS_EXTEND_FORWARD - mask_tensor.shape[2]
        # assert mask_years_missing == 0
        sat_tensor = mask_image(sat_tensor, mask_tensor, 8 , mask_years_missing, NUM_YEARS_EXTEND_FORWARD,num_imgs_per_year)

        if not indices_only:
            temp_tensor = mask_image(temp_tensor, mask_tensor, NUM_TEMP_BANDS, mask_years_missing, NUM_YEARS_EXTEND_FORWARD,num_imgs_per_year)
            sat_histograms = calc_histograms(sat_tensor, HIST_BINS_LIST[:NUM_REF_BANDS]) 
            temp_histograms = calc_histograms(temp_tensor,HIST_BINS_LIST[NUM_REF_BANDS:NUM_REF_BANDS+NUM_TEMP_BANDS])
            histograms = np.concatenate((sat_histograms, temp_histograms), axis=2)
            # print(histograms.shape[1],len(dates))
            assert len(dates) == histograms.shape[1]
            reshaped_dates = np.array(dates).reshape(1, len(dates), 1)
            reshaped_dates = np.repeat(reshaped_dates, histograms.shape[0], axis=0)
            histograms_dates = np.concatenate((histograms, reshaped_dates), axis=2)
            np.save(hist_save_path, histograms_dates)

        
        count += 1
        print("{} {}/{}".format(place, count, num_tifs))
    
    print('Generated', count, 'histogram' if count == 1 else 'histograms')
    # Now we generate the plots. 
    # plots_directory = join(target_folder_name,'Plots')
    # if not isdir(plots_directory): mkdir(plots_directory)
    # analyze_histograms(target_folder_name, plots_directory)

