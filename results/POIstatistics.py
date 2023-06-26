import os.path

import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
import skimage
from skimage.feature import peak_local_max
from scipy.spatial import distance
from pathlib import Path

# #NOT WORKING, OUTDATED
# def findLocalMaxima(fname, plot=False, neighborhood_size=5, threshold=1500):
#
#     data = scipy.misc.imread(fname) #read image
#
#     data_max = filters.maximum_filter(data, neighborhood_size) #find local max
#     maxima = (data == data_max)
#     data_min = filters.minimum_filter(data, neighborhood_size) #find local minima
#     diff = ((data_max - data_min) > threshold)
#     maxima[diff == 0] = 0 #remove local min from local max?
#
#     labeled, num_objects = ndimage.label(maxima)
#     slices = ndimage.find_objects(labeled)
#     x, y = [], []
#     for dy,dx in slices: #find centers of local max
#         x_center = (dx.start + dx.stop - 1)/2
#         x.append(x_center)
#         y_center = (dy.start + dy.stop - 1)/2
#         y.append(y_center)
#
#     if plot:
#         plt.imshow(data)
#         #plt.savefig('/tmp/data.png', bbox_inches = 'tight')
#
#         plt.autoscale(False)
#         plt.plot(x,y, 'ro')
#         #plt.savefig('/tmp/result.png', bbox_inches = 'tight')
#
#     return x, y

def iterate(folder_path, folder_path2, folder_path3):
    files1 = Path(folder_path).glob('*')
    files2 = Path(folder_path2).glob('*')
    files3 = Path(folder_path3).glob('*')
    for fhs in zip(files1, files2, files3):
        yield fhs


if __name__ == '__main__':

    datapath = 'analysis/peaks_pix2pix_3channels_normalized'
    datapathGen = os.path.join(datapath, 'gen')
    datapathGT = os.path.join(datapath, 'GT')
    datapathDEM = os.path.join(datapath, 'DEM')
    outpath = os.path.join('./analysis',datapath)

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    accuracy = []
    imagename= []
    # similarityNumber = []
    for filegt, filegen, filedem in iterate(datapathGT, datapathGen, datapathDEM):
        #x, y = findLocalMaxima(os.path.join(datapath, filename))
        # gtpeaks = skimage.io.imread(os.path.join(datapathGT, filegt), as_gray=True)
        gtpeaks = skimage.io.imread(filegt, as_gray=True)
        # genpeaks = skimage.io.imread(os.path.join(datapathGT, filegt), as_gray=True)
        genpeaks = skimage.io.imread(filegen, as_gray=True)
        #find peaks (or POIs) in the GT data
        xygt = peak_local_max(gtpeaks, min_distance=2, exclude_border=False)
        xygen = peak_local_max(genpeaks, min_distance=int(len(gtpeaks)*0.02), threshold_rel=0.5, exclude_border=False)
        xygen_orig = xygen
        #for each gt POI
        #for peak in xygt:
            #see if there is a close enough generated poi
        dist_matrix = distance.cdist(xygt, xygen, 'euclidean')
        r = len(gtpeaks)*0.05 #we allow for a deviation of a 5% of the total image size
        #result = np.count_nonzero(dist_matrix <= r)
        bool_matrix = dist_matrix <= r
        result_mid = np.asarray(np.where(bool_matrix)) #indexes of matches of close peaks
        result = []
        for i in range(len(xygt)):
            tmp = result_mid[0,:] == i
            if np.count_nonzero(tmp) == 0:
                continue
            elif np.count_nonzero(tmp) == 1:
                result.append(result_mid[:, tmp])
            elif np.count_nonzero(tmp) > 1: #keep only the closest generated peak to the GT
                candidates = result_mid[:, tmp]
                mindistidx = np.argmin(dist_matrix[candidates[0, :], candidates[1, :]])
                result.append(candidates[:, mindistidx])
        tp = len(result) #true positives
        fp = len(xygen) - len(result) #false positives
        fn = len(xygt) - len(result) #false negatives
        accuracy.append(len(result)/len(xygt))
        imagename.append([filegt, filegen])
        # similarityNumber.append(np.abs((len(result) - len(xygt)) / len(xygt)))
        # print(tp, fp, fn, accuracy)

        if((len(result)/len(xygt))<0.5):
            # subplot(r,c) provide the no. of rows and columns
            f, axarr = plt.subplots(1, 2)
            gtdem = skimage.io.imread(filedem, as_gray=True)
            alphas = np.ones(gtdem.shape)*0.25
            # use the created array to output your multiple images. In this case I have stacked 4 images vertically
            axarr[0].imshow(gtdem)
            axarr[0].imshow(gtpeaks, alpha=alphas)
            axarr[0].plot(xygt[:,1], xygt[:,0],'r.')
            axarr[0].set_title('GT peaks')
            axarr[1].imshow(gtdem)
            axarr[1].imshow(genpeaks, alpha=alphas)
            axarr[1].plot(xygen_orig[:, 1], xygen_orig[:, 0], 'g.')
            axarr[1].plot(xygen[:, 1], xygen[:, 0],'r.')
            axarr[1].set_title('Gen peaks')
            plt.show()


    print('mean accuracy: ', np.mean(np.asarray(accuracy)))
    plt.figure()
    plt.hist(accuracy)
    plt.show()


    # print('mean deviation in number of generated peaks: ', np.mean(np.asarray(similarityNumber)))

        #TODO: check que nos estamos guardando las distancias wrt al GT
        #TODO: get accuracy now that the gen peak local max detection works well