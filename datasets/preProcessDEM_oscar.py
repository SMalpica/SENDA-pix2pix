import os

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from osgeo import gdal

from scipy.signal import convolve, convolve2d
from scipy.ndimage import gaussian_filter, minimum_filter, maximum_filter
import png

import richdem as rd  # https://richdem.readthedocs.io/


def savePNG(filename, values, b=16, normalize=True, grayscale=True):
    x = values.astype(np.uint16)
    if normalize:
        x = (((1 << b) - 1) * (values - values.min()) / (values.max() - values.min())).astype(np.uint16)
    # else:
    #    x = (((1 << b) - 1) * (values)).astype(int)
    # Convert y to 16 bit unsigned integers.
    # else:
    #     x = (65535 * ((x - x.min()) / x.ptp())).astype(np.uint16)
    with open(filename, 'wb') as f:

        writer = png.Writer(width=x.shape[1], height=x.shape[0], bitdepth=b, greyscale=grayscale)
        # Convert z to the Python list of lists expected by
        # the png writer.
        z2list = x.reshape(-1, x.shape[1] * x.shape[2]).tolist()
        writer.write(f, z2list)
        # writer.write(f, x.tolist())


# # normal of a heightfield h: N(h(x,y)) = (dh/dx, dh/dy, 1)
def Normals(elevs):
    # Nx = np.gradient(elevs, cellSize, axis=1)
    # Ny = np.gradient(elevs, cellSize, axis=0)
    Nx = np.gradient(elevs, axis=1)
    Ny = np.gradient(elevs, axis=0)
    N = np.dstack([-Nx, Ny, np.ones(elevs.shape)]) # keep Ny positive because Y axis is image oriented (pointing south)
    N = N/np.linalg.norm(N, axis=2, keepdims=True)
    return N

def GradientNorm(elevs):
    # Nx = np.gradient(elevs, cellSize, axis=1)
    # Ny = np.gradient(elevs, cellSize, axis=0)
    Nx = np.gradient(elevs, axis=1)
    Ny = np.gradient(elevs, axis=0)
    return np.sqrt(Nx*Nx + Ny*Ny)

def SlopeAngle(elevs):
    return np.arctan(GradientNorm(elevs))

# curvatures See "[Schmidt 2003] - Comparison of polynomial models for land surface curvature calculation" and "[
# Shary 2002] Fundamental quantitative methods of land surface analysis" for a summary of formulations. Note that
# Table 1 in Schmidt contains a typo for contour curvature.
def Curvatures(elevs):
    # fx = np.gradient(elevs, cellSize,  axis=1)
    # fy = np.gradient(elevs, cellSize, axis=0)
    # fxx = np.gradient(fx, cellSize, axis=1)
    # fyy = np.gradient(fy, cellSize, axis=0)
    # fxy = np.gradient(fx, cellSize, axis=0)
    fx = np.gradient(elevs, axis=1)
    fy = np.gradient(elevs, axis=0)
    fxx = np.gradient(fx, axis=1)
    fyy = np.gradient(fy, axis=0)
    fxy = np.gradient(fx, axis=0)

    Kp = -(fxx * fx * fx + 2 * fxy * fx * fy + fyy * fy * fy)
    Kc = -(fxx * fy * fy - 2 * fxy * fx * fy + fyy * fx * fx)
    Kt = -(fxx * fy * fy - 2 * fxy * fx * fy + fyy * fx * fx)
    d = fx * fx + fy * fy
    Kpd = d * np.power(d + 1, 3 / 2)
    Kcd = np.power(d, 3 / 2)
    Ktd = d * np.power(d + 1, 1 / 2)

    Kprofile = np.zeros(elevs.shape)
    Kcontour = np.zeros(elevs.shape)
    Ktangent = np.zeros(elevs.shape)
    Kprofile[d > 0] = Kp[d > 0] / Kpd[d > 0]
    Kcontour[d > 0] = Kc[d > 0] / Kcd[d > 0]
    Ktangent[d > 0] = Kt[d > 0] / Ktd[d > 0]

    return Kprofile, Kcontour, Ktangent

# Following Dikau 1989 classification using a 3x3 matrix from tangential and profile concavity/convexity.
def DikauCurvatureLandforms(Kt, Kp, t=0.1):
    vt = np.ones(Kt.shape)
    vt[Kt < -t] = 0  # convex
    vt[Kt > t] = 2  # concave

    vp = np.ones(Kp.shape)
    vp[Kp < -t] = 0  # convex
    vp[Kp > t] = 2  # concave

    # Classification according to Profile-Tangential curvatures
    #   0: convex-convex     -> nose
    #   1: convex-straight   -> shoulder slope
    #   2: convex-concave    -> hollow shoulder
    #   3: straight-convex   -> spur
    #   4: straight-straight -> planar slope
    #   5: straight-concave  -> hollow
    #   6: concave-convex    -> spur foot
    #   7: concave-straight  -> foot slope
    #   8: concave-concave   -> hollow foot

    return (3 * vp + vt).astype(int)


# Topographic Position Index
# How much a point rises with respect to the mean elevation of its neighborhood.
#
# Highlights ridges (>0) and rivers (<0).
def TPI(elevs, w):
    # central point minus mean of (w^2 - 1) neighbors
    kernel = -1.0/(w*w - 1) * np.ones((w, w))
    kernel[(w-1)//2, (w-1)//2] = 1
    return convolve2d(elevs, kernel, mode='same', boundary='symm')

# Local Variance
# Local variance is defined as the standard deviation of a window centered around the cell.
def LocalVariance(elevs, w):
    n  = w*w
    s  = convolve2d(elevs, np.ones((w, w)), mode='same', boundary='symm')
    ss = convolve2d(elevs*elevs, np.ones((w, w)), mode='same', boundary='symm')
    var = (n*ss - s*s)/(n*(n-1))
    return np.sqrt(var)

#
# Surface Roughness
# Lindsay et al. 2019. Scale-optimized surface roughness for topographic analysis
#
# The authors associate surface normals dispersion to roughness, a measure of texture complexity. In particular,
# they measure the spherical standard deviation in a window  𝑤×𝑤 , i.e. the angular spread of the normal directions
# in this window. It will be 0 for flat and inclined planes, and increase with surface texture complexity.
def SurfaceRoughness(elevs, w, blur=False):
    r = w // 2

    # low-pass filter to remove higher frequencies (smaller-scale details)
    # this way, we obtain a signature that depends on current w scale
    if blur:
        els = gaussian_filter(elevs, r)
    else:
        els = elevs

    # normal map
    N = Normals(els)

    # create a Summed Area Table to optimize finding the average normal in a region
    SAT = np.zeros(N.shape)
    SAT[0:1, :, :] = np.cumsum(N[0:1, :, :], axis=1)
    SAT[:, 0:1, :] = np.cumsum(N[:, 0:1, :], axis=0)
    for i in range(1, N.shape[0]):
        for j in range(1, N.shape[1]):
            SAT[i, j] = N[i, j] + SAT[i - 1, j] + SAT[i, j - 1] - SAT[i - 1, j - 1]

    # sum of normals inside w x w window
    # note: this code is vectorized. See below for loop version to better understand it
    imin = np.maximum(0, np.arange(N.shape[0]) - r)
    jmin = np.maximum(0, np.arange(N.shape[1]) - r)
    imax = np.concatenate([np.arange(r, N.shape[0]), np.full((r,), N.shape[0] - 1)])
    jmax = np.concatenate([np.arange(r, N.shape[1]), np.full((r,), N.shape[1] - 1)])
    iimin, jjmin = np.meshgrid(imin, jmin, indexing='ij')
    iimax, jjmax = np.meshgrid(imax, jmax, indexing='ij')

    Nsums = SAT[iimax, jjmax, :]
    Nsums[r + 1:, :] -= (SAT[iimin - 1, jjmax, :])[r + 1:, :, :]
    Nsums[:, r + 1:] -= (SAT[iimax, jjmin - 1, :])[:, r + 1:, :]
    Nsums[r + 1:, r + 1:] += (SAT[iimin - 1, jjmin - 1, :])[r + 1:, r + 1:, :]

    Ncnts = (iimax - iimin + 1) * (jjmax - jjmin + 1)

    # roughness
    R = np.linalg.norm(Nsums, axis=2)
    Rn = np.minimum(R / Ncnts, 1)  # clip to 1 to avoid for precision errors
    rough = np.sqrt(-2.0 * np.log(Rn)) * 180.0 / np.pi
    return rough


# # Loop version for the code above using the SAT
# NsumsCheck = np.zeros(N.shape)
# NcntsCheck = np.zeros(N.shape)[:,:,0]
# for i in range(N.shape[0]):
#     for j in range(N.shape[1]):
#         imin = max(i-r, 0)
#         imax = min(i+r, N.shape[0] - 1)
#         jmin = max(j-r, 0)
#         jmax = min(j+r, N.shape[1] - 1)
#         NsumsCheck[i,j] = SAT[imax,jmax]
#         if imin > 0:
#             NsumsCheck[i,j] -= SAT[imin-1,jmax]
#         if jmin > 0:
#             NsumsCheck[i,j] -= SAT[imax, jmin-1]
#         if imin > 0 and jmin > 0:
#             NsumsCheck[i,j] += SAT[imin-1, jmin-1]
#         NcntsCheck[i,j] = (imax - imin + 1)*(jmax - jmin + 1)

#
# Rivers: Stream Area and Wetness Index
# Stream area of a point p is the area of its drainage basin, i.e. the area of
# all upstream points from which flow reaches p.
#
# To compute it, we start at the highest terrain point and distribute a flow unit to its 8 neighbors proportionally
# to each gradient direction.
#
# The result can be used as a river detector, more on that later. Also, for better outputs, a Breaching algorithm
# might be needed before, to account for DEM precision errors that "cut" river heights decreasing monotonicity.


#TODO: script con las funciones de las métricas para importar desde mis programas
if __name__ == '__main__':

    datafolder = 'train'
    outfolder = 'normalized'

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    for filename in os.listdir(datafolder):

        print(filename)
        

        plot = False
        # plot = False



        dem = png.Reader(os.path.join(datafolder,filename))  # creates data reader
        width, height, demreader, info = dem.read()  # reads metadata and iterator
        elevGrid = np.vstack(map(np.uint16, demreader))  # uses an iterator to create a 2D array, row by row
        elevGrid = elevGrid.astype(float)
        # not a problem in our case because our data is supposed to be grayscale
        # to reshape to 3D
        # image_3d = numpy.reshape(image_2d, (row_count, column_count, plane_count))
        elevGrid[elevGrid < 0] = 0

        # limits for debug visualizations
        pimin = 0
        pimax = elevGrid.shape[0]
        pjmin = 0
        pjmax = elevGrid.shape[1]
        #
        # # if False:  # note: set to false to process full dataset
        # #     elevGrid = elevGrid[pimin:pimax, pjmin:pjmax]
        # #     pimax -= pimin
        # #     pimin = 0
        # #     pjmax -= pjmin
        # #     pjmin = 0
        #
        # if plot==True:
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(elevGrid[pimin:pimax, pjmin:pjmax], cmap='terrain')

        # N = Normals(elevGrid) #normals of the elevation map
        # slopeAngle = SlopeAngle(elevGrid) #slope angle
        #
        # if plot==True:
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(N * 0.5 + 0.5)
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(slopeAngle, vmin=0, vmax=np.pi * 0.5, cmap='coolwarm')
        #     # savePNG('slopeAngle.png', (slopeAngle * 180 / np.pi) / 90.0, 8)
        #     plt.hist(SlopeAngle(elevGrid).flatten() * 180 / np.pi)

        # Kprofile, Kcontour, Ktangent = Curvatures(elevGrid) #curvatures, contours, tangents
        #
        # if plot==True:
        #     fig = plt.figure(figsize=(30, 10))
        #     ax = fig.add_subplot(131)
        #     ax.imshow(Kprofile, vmin=-0.1, vmax=0.1, cmap='coolwarm')
        #     ax = fig.add_subplot(132)
        #     ax.imshow(Kcontour, vmin=-0.2, vmax=0.2, cmap='coolwarm')
        #     ax = fig.add_subplot(133)
        #     ax.imshow(Ktangent, vmin=-0.1, vmax=0.1, cmap='coolwarm')

        # Dikau = DikauCurvatureLandforms(Ktangent, Kprofile, t=0.05)
        #
        # if plot==True:
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(Dikau, cmap='Set1')

        # tpi = TPI(elevGrid, 5)
        # if plot == True:
        #     print('Percentile 0.1%:', np.percentile(tpi, 0.1))
        #     print('Percentile 99.9%:', np.percentile(tpi, 99.9))
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(tpi[pimin:pimax, pjmin:pjmax], vmin=-4, vmax=4, cmap='coolwarm')
        #     plt.show()

        # tpi = TPI(elevGrid, 10)
        # if plot == True:
        #     print('Percentile 0.1%:', np.percentile(tpi, 0.1))
        #     print('Percentile 99.9%:', np.percentile(tpi, 99.9))
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(tpi[pimin:pimax, pjmin:pjmax], vmin=-4, vmax=4, cmap='coolwarm')
        #     plt.show()

        tpi = TPI(elevGrid, 15)
        tpi = 0.5 + 0.5*tpi/(3*tpi.std())
        tpi = np.minimum(1, np.maximum(0, tpi))
        if plot == True:
            print('Percentile 0.1%:', np.percentile(tpi, 0.1))
            print('Percentile 99.9%:', np.percentile(tpi, 99.9))
            plt.figure(figsize=(10, 10))
            plt.imshow(tpi[pimin:pimax, pjmin:pjmax], vmin=0, vmax=1, cmap='coolwarm')
            plt.show()

        # tpi = TPI(elevGrid, 29)
        # if plot == True:
        #     print('Percentile 0.1%:', np.percentile(tpi, 0.1))
        #     print('Percentile 99.9%:', np.percentile(tpi, 99.9))
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(tpi[pimin:pimax, pjmin:pjmax], vmin=-15, vmax=15, cmap='coolwarm')

        #rivers stream area and wetness index (using richdem library for optimization)
        # breach DEM
        #elevGridBreached = rd.BreachDepressions(rd.rdarray(elevGrid, no_data=-9999))

        # apply an epsilon fill to convert flat areas to inclined planes
        # this is needed because breaching algorithm stops at equal height, but flow needs some slope
        elevGridBreached = rd.FillDepressions(rd.rdarray(elevGrid, no_data=-9999), epsilon=True)

        # compute flow using Dinf algorithm
        saRD = rd.FlowAccumulation(elevGridBreached, method='Quinn')

        if plot==True:
            plt.figure(figsize=(20, 20))
            plt.imshow(saRD[pimin:pimax, pjmin:pjmax], cmap='coolwarm',
                   norm=mcolors.LogNorm(vmin=saRD.min(), vmax=np.percentile(saRD, 99.5)))
            plt.show()

        #Wetness Index = Stream Area / Slope
        c = 1.0
        slope = GradientNorm(elevGrid)
        wi = np.log(saRD / (1 + c * slope))
        wi_min = np.percentile(wi, 0.1)
        wi_max = np.percentile(wi, 99.9)
        wi = (wi - wi_min)/(wi_max - wi_min)
        wi = np.minimum(1, np.maximum(0, wi))

        if plot==True:
            fig = plt.figure(figsize=(20, 20))
            plt.imshow(wi[pimin:pimax, pjmin:pjmax], cmap='coolwarm', vmin=0, vmax=1)
            plt.show()

        dem3channels = np.stack((elevGrid, (tpi * 65535).astype(np.uint16), (wi * 65535).astype(np.uint16)), axis=2)
        savePNG(os.path.join(outfolder, filename), dem3channels, normalize=False, grayscale=False)
        