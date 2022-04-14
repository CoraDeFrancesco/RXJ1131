# RXJ1131
 GL analysis

Requires magnification maps. :) 

Takes Kerr images, convolves with magnification maps of images A-D.

A number of random tracks are drawn across the 4 images, and the track light curves are added for each image for each simulated Kerr black hole.

This results in a matrix with dimensions (number of kerr images, number of tracks) consisting of light curves representing the Kerr image traveling across the foreground galaxy.

One Kerr image is selected as the base Kerr image. All the light curves are divided by the corresponding track light curve for the base image.

Finally, the measured change in reflection scalar is compared with the divided light curves (resampled to have the same time resolution as the data). 

The $\chi ^2$ is calculated for each track, and the minimum $\chi ^2$ track is plotted against the data.
