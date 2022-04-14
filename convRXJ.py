#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 09:28:17 2022

@author: user1
"""

from scipy import signal
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import chisquare

#%% Load in Kerr image

imagefile1 = 'mag_map/Image_I_a0.9_inc40_nr5.8_Gamma1.0.dat'
imagefile2 = 'mag_map/Image_I_a0.9_inc40_nr2.6.dat'
image1 = np.loadtxt(imagefile1)
image2 = np.loadtxt(imagefile2)
print('Kerr images loaded.')

kerr_images = [image1, image2]

#%% Linearize and rebin Kerr image

image_rebins = []

for image in kerr_images:
    
    image_lin = 10.**image
    # image_trim = image_lin[1:400, 1:400]
    # I think the above slice was from the bottom left corner of the images.
    #   Now taking the same size slice from the center.
    image_trim = image_lin[2000-199:2000+200, 2000-199:2000+200]
    
    image_rebin_raw = np.reshape(image_trim, (133,3,133,3)).mean(-1).mean(1)
    image_rebin = image_rebin_raw/np.sum(image_rebin_raw)
    
    image_rebins.append(image_rebin)
    

# image_lin = 10.**image
# image_trim = image_lin[1:400, 1:400]

# image_rebin_raw = np.reshape(image_trim, (133,3,133,3)).mean(-1).mean(1)
# image_rebin = image_rebin_raw/np.sum(image_rebin_raw)
print('Kerr images rebinned.')

#%% Load in mag map


magn_map_A = 'mag_map/1131A_pattern_binary64'

magn_map_B = 'mag_map/1131B_pattern_binary64'

magn_map_C = 'mag_map/1131C_pattern_binary64'

magn_map_D = 'mag_map/1131D_pattern_binary64'

dt = np.dtype([('pad1', np.int32), ('value', np.float64), ('pad2', np.int32)])

t_start1   = time.time()

print('Reading in mag map A.')
magmapA = np.memmap(magn_map_A, dtype=dt, mode='r', shape=16000*16000)
magmapA = magmapA['value'].reshape(16000, 16000)
print('Reading in mag map B.')
magmapB = np.memmap(magn_map_B, dtype=dt, mode='r', shape=16000*16000)
magmapB = magmapB['value'].reshape(16000, 16000)
print('Reading in mag map C.')
magmapC = np.memmap(magn_map_C, dtype=dt, mode='r', shape=16000*16000)
magmapC = magmapC['value'].reshape(16000, 16000)
print('Reading in mag map D.')
magmapD = np.memmap(magn_map_D, dtype=dt, mode='r', shape=16000*16000)
magmapD = magmapD['value'].reshape(16000, 16000)

print('Reshaping mag maps.')

t_finish1  = time.time()
read_time  = t_finish1 - t_start1         # in seconds

print('read_time:', read_time)

#%% Look at magmaps

fig, axs = plt.subplots(2, 2, dpi=100, figsize=(10,10))
axs[0, 0].imshow(np.log(magmapA))
axs[0, 0].set_title('Mag map A')

axs[0, 1].imshow(np.log(magmapB))
axs[0, 1].set_title('Mag map B')

axs[1, 0].imshow(np.log(magmapC))
axs[1, 0].set_title('Mag map C')

axs[1, 1].imshow(np.log(magmapD))
axs[1, 1].set_title('Mag map D')

for ax in axs.flat:
    ax.label_outer()

plt.show()
plt.clf()


#%% Look at kerr images

for i, kerr_image in enumerate(kerr_images):
    plt.figure(dpi=100)
    plt.imshow(kerr_image)
    plt.title(('Kerr Image ' + str(i+1)))
    plt.show()
    plt.clf()
    
plt.figure(dpi=100)
plt.imshow((image_rebins[0] - image_rebins[1]))
plt.colorbar()
plt.title('Kerr Image Dif')
plt.show()
plt.clf()    

#%% Convolution

print('Beginning convolution.')
t_start = time.time()

magconvs_images = []

for i, image_rebin in enumerate(image_rebins):
    
    print('    Convolution for kerr image', (i+1))
    
    magconvs_image = []
    
    print('        Image A...')
    magconvA = signal.fftconvolve(magmapA, image_rebin, mode='same')
    print('        Image B...')
    magconvB = signal.fftconvolve(magmapB, image_rebin, mode='same')
    print('        Image C...')
    magconvC = signal.fftconvolve(magmapC, image_rebin, mode='same')
    print('        Image D...')
    magconvD = signal.fftconvolve(magmapD, image_rebin, mode='same')
    
    magconvs_image.append(magconvA)
    magconvs_image.append(magconvB)
    magconvs_image.append(magconvC)
    magconvs_image.append(magconvD)
    
    magconvs_images.append(magconvs_image)

t_end = time.time()
print('Finished colvolution.')
print('runtime:', (t_end - t_start))


#%% Setup Track and Light Curve

# pick two random points in the convolved image.

mag_shape = magconvs_images[0][0].shape # Assuming all images have the same shape

def gen_track():
    
    x1,x2 = np.random.randint(low=0, high=mag_shape[0], size=2)
    y1,y2 = np.random.randint(low=0, high=mag_shape[0], size=2)
    
    # generate a line between the points
    
    m = (y2-y1) / (x2-x1)
    b = y1- (m*x1)
    
    x_arr = np.linspace(x1, x2, num=(abs(x2-x1)+1))
    y_arr = (m*x_arr) + b
    
    # convert to years
    
    R_E_map = 2.13
    pix_map = 16040
    t_E = 11.13 #yr
    
    yr_per_pix = (R_E_map / pix_map) * t_E
    
    time_track = x_arr * yr_per_pix # plot vs time instead of pixels
    ini_time = time_track[0] # adjust time to start at 0
    time_track = time_track - ini_time
    
    return(x_arr, y_arr, time_track)
    

#%% Generate several random tracks

R_E_map = 2.13
pix_map = 16040
t_E = 11.13 #yr
time_cut = 2.5
pix_cut = int(time_cut*677) #pixels in 3 years

yr_per_pix = (R_E_map / pix_map) * t_E

num_tracks = 10 # how many tracks to generate

save_x = []
save_y = []
save_time = []

# Keep only tracks with max time over 2 years

while (len(save_x) < num_tracks):
    
    x_arr, y_arr, time_arr = gen_track()
    
    while (max(time_arr) < time_cut): # Make sure all trakcs are > time_cut
                                        
        print('Max time of tracks = ', max(time_arr))
        print('Redrawing to get max > ', time_cut, 'years.')
        
        x_arr, y_arr, time_arr = gen_track()
        
    
    if (max(time_arr) >= time_cut):
        
        #trim and append
        
        x_new = np.asarray(x_arr[:pix_cut])
        y_new = np.asarray(y_arr[:pix_cut])
        time_new = np.asarray(time_arr[:pix_cut])
        
        save_x.append(x_new)
        save_y.append(y_new)
        save_time.append(time_new)
        

#%% Magvals for each track

# grab the magnification values for each point on the line

def get_magvals(magconv, x_arr):
    
    dummy_track = []
    for i,x in enumerate(x_arr):
        x = int(x)
        y = int(y_arr[i])
        mag_val = magconv[x, y]
        dummy_track.append(mag_val)
    mag_track = np.asarray(dummy_track)
    
    return(mag_track)

def get_tracks_brightness(magconvs_images, x_track, y_track):
    
    magtracks = [] # place to store brightness values for track
    
    # loop through kerr images
    for i, kerr_im in enumerate(magconvs_images):
        print('Finding tracks for kerr_im', (i+1))
        kerr_tracks = []
        for j, magconv in enumerate(kerr_im): # loop through A-D and get brightness
            print('... image', (j+1))
            magvals = get_magvals(magconv, x_track)
            kerr_tracks.append(magvals)
        magtracks.append(kerr_tracks)
        
    return(magtracks)

# test get_tracks_brightness()

# track_idx = 3

# magtracks = get_tracks_brightness(magconvs_images, save_x[track_idx], save_y[track_idx])


#%% Light curve vs time

# R_E_map = 2.13
# pix_map = 16040
# t_E = 11.13 #yr

# yr_per_pix = (R_E_map / pix_map) * t_E

# time_track = x_arr * yr_per_pix # plot vs time instead of pixels
# ini_time = min(time_track) # adjust time to start at 0
# time_track = time_track - ini_time

def get_total_track(magtracks, kerr_im_to_plot):
    
    total_mag_track = magtracks[kerr_im_to_plot][0] + magtracks[kerr_im_to_plot][1] + \
        magtracks[kerr_im_to_plot][2] + magtracks[kerr_im_to_plot][3]
        
    return(total_mag_track)

total_tracks = []

for i in range(len(kerr_images)): # loop through kerr images
    
    tracks_for_im_i = []    
    
    for j in range(len(save_x)): # loop through tracks
    
        # brightness for each image with this track
        magtracks = get_tracks_brightness(magconvs_images, save_x[j], save_y[j])
        
        # total brightness for source
        tracks_for_im_i.append(np.asarray(get_total_track(magtracks, i)))
        
    total_tracks.append(np.asarray(tracks_for_im_i))
    

#%% Plot selected image to check

kerr_im_to_plot = 0 # 0 or 1
mag_track_to_plot = 3 # 0 to 3 for images A through D
track_num_to_plot = 8 # 0 to 9 (for now)

image_letters = ['A', 'B', 'C', 'D']
image_letter = image_letters[mag_track_to_plot]


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,6))

ax1.imshow(magconvs_images[kerr_im_to_plot][mag_track_to_plot])
#ax1.plot(x_arr, y_arr, color='red')
ax1.plot(save_x[track_num_to_plot], save_y[track_num_to_plot], color='red')
ax1.set_title(('Convolved Image ' + str(image_letter)+ ' Track '+ str(track_num_to_plot)))

ax2.plot(save_time[track_num_to_plot],magtracks[kerr_im_to_plot][mag_track_to_plot], color='red')
ax2.set_xlabel('Time (years)')
ax2.set_ylabel('Brightness')
ax2.set_title(('Theoretical Light Curve '+ image_letter + ' Kerr Im '+ \
               str(kerr_im_to_plot+1)))

plt.show()

plt.clf()    

plt.figure(dpi=100)
plt.plot(save_time[track_num_to_plot], total_tracks[kerr_im_to_plot][track_num_to_plot], color='red')
plt.xlabel('Time (years)')
plt.ylabel('Brightness')
plt.title(('Total Theoretical Light Curve, Kerr Image ' +str(kerr_im_to_plot +1)))
plt.show()
plt.clf()

#%% Load Data

# Load in FPMA, FPMB light curves

comb_data = np.loadtxt("data_tables/comb_fits_results.csv", skiprows=1, delimiter=',').T

date = comb_data[0]
jd = comb_data[1]
time_dat = (jd - min(jd)) * (1/365)
r_refl = comb_data[2]
r_refl_lower = comb_data[3]
r_refl_upper = comb_data[4]
norm = comb_data[5]
norm_lower = comb_data[6]
norm_upper = comb_data[7]

# create the error arrays 
# note that the lower error still must be in positive values
# but is given as negative values in the data

r_refl_error = [(-r_refl_lower), (r_refl_upper)]

# Check

plt.figure(dpi=200)

plt.plot(time_dat, r_refl, marker='o')

# label = ('Total Theoretical Light Curve, Kerr Image ' \
#          +str(kerr_im_to_plot +1) + ' Track ' + str(track_num_to_plot +1))

# plt.plot(save_time[track_num_to_plot], (0.5e-12)*total_tracks[kerr_im_to_plot][track_num_to_plot], \
#          color='red', label=label)

plt.xlabel('Time (years)')
plt.ylabel('Reflection Scalar')
plt.title('Data')

plt.show()
plt.clf()

#%% Adjust Res to Data

# Adjust theoretical light cuves to have same time resolution as data.

test_data = r_refl
test_data_time= time_dat

dtime=10/365 # delta time to search for match = 1 day

# Resample the theoretical tracks to have same number of time bins as data.

sampled_tracks = []
sampled_times = []

for i in range(len(kerr_images)): # loop through kerr images
    
    tracks_for_im_i = []
    times_for_im_i = []
    #print('Inside image', i)
    
    for j in range(len(save_x)): # loop through tracks
    
        #print('Inside track', j)
        
        # Set up array for resampled values
        resampled_track = np.zeros(len(test_data_time))
        resampled_time = np.zeros(len(test_data_time))
        
        # Total LC for kerr i, track j
        current_track_brightness = total_tracks[i][j]
        current_track_time = save_time[j]
        

    
        
        for k, time_val in enumerate(test_data_time): # loop through times
            
            ltime= time_val-dtime
            utime= time_val+dtime
            
            #print('Bounds for time', k, ':', ltime, utime)
            
            
            time_mask = np.where((current_track_time >= ltime) & \
                                 (current_track_time <= utime))
            idx = int(np.median(time_mask))
            
            #print('   Using idx', idx)
            
            resampled_track[k] = current_track_brightness[idx]
            resampled_time[k] = current_track_time[idx]
        
        
        # total brightness for source
        #print('Track', j, 'successful!')
        tracks_for_im_i.append(resampled_track)
        times_for_im_i.append(resampled_time)
        
    sampled_tracks.append(np.asarray(tracks_for_im_i))
    sampled_times.append(np.asarray(times_for_im_i))
    
#%% Divided tracks

base_kerr_im = 0 # Kerr im idx by which to divide the rest

div_tracks = []

for i in range(len(kerr_images)):
    
    tracks_for_im_i = []
    
    for j in range(len(save_x)):
        
        div_track = sampled_tracks[i][j] / sampled_tracks[base_kerr_im][j]
        
        tracks_for_im_i.append(div_track)
        
    div_tracks.append(tracks_for_im_i)    
    
# Check

plt.figure(dpi=200)

plt.plot(time_dat, r_refl, label='FPMA', marker='o')

label = ('Divided Tracks, Kerr Image ' \
         +str(kerr_im_to_plot +1) + ' Track ' + str(track_num_to_plot +1))

plt.plot(sampled_times[kerr_im_to_plot][track_num_to_plot], div_tracks[kerr_im_to_plot][track_num_to_plot], \
         color='red', label=label, marker='o')

plt.xlabel('Time (years)')
plt.ylabel('r_refl')

plt.legend(fontsize=6)

plt.show()
plt.clf()

#%% Calculate X2
    
x2_vals = []

for i in range(len(kerr_images)): # loop through kerr images
    
    x2_for_im_i = []
    
    for j in range(len(save_x)): # loop through tracks
        
        x2_val = chisquare(div_tracks[i][j], test_data)[0] # grabs just x2, not p
        
        x2_for_im_i.append(x2_val)
        
    x2_vals.append(np.asarray(x2_for_im_i))

# Find track corresponsing to min X2

min_mask = np.where(x2_vals == np.min(x2_vals))
kerr_min_idx = min_mask[0][0]
track_min_idx = min_mask[1][0]

# Plot Min Theoretical Track

div_min = total_tracks[kerr_min_idx][track_min_idx] / total_tracks[base_kerr_im][track_min_idx]

# Check

plt.figure(dpi=200)

plt.plot(time_dat, r_refl, label='R_refl Data', marker='o')

label = ('Theoretical Ratio, Kerr Image ' \
         +str(kerr_min_idx +1) + ' Track ' + str(track_min_idx +1))
    
plt.plot(save_time[track_min_idx], \
          div_min, \
          color='red', label=label, alpha=0.5)

plt.plot(sampled_times[kerr_min_idx][track_min_idx], \
         div_tracks[kerr_min_idx][track_min_idx], \
         color='red', label='Resampled Theoretical Ratio', marker='o')

plt.xlabel('Time (years)')
plt.ylabel('r_refl')
plt.title('Minimum X^2 Refl Scalar')

plt.legend(fontsize=6)

plt.show()
plt.clf()
