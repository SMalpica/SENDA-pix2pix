import os
import re

'''
pix2pix needs images from category A and from category B to be named the same in order to run the combina_a_and_b script. 
This script will rename them from A <DEM> "elevs_xxx.png" and B <POIPeaks> "xxx_alps_peaksOnly.png" to "xxx.png"
'''

# Function to rename multiple files
def rename(path):
    i = 0

    for filename in os.listdir(path):
        my_dest =re.findall(r'\d+', filename)
        my_source = os.path.join(path, filename)
        my_dest = my_dest[0] + '.png'
        my_dest = os.path.join(path, my_dest)
        # rename() function will
        # rename all the files
        os.rename(my_source, my_dest)
        i += 1
# Driver Code
if __name__ == '__main__':
    # path = "./water/DEM/test"
    # rename(path)
    # path = "./water/DEM/train"
    # rename(path)
    # path = "./water/DEM/val"
    # rename(path)
    path = "./geopoi/POIgeo/test"
    rename(path)
    path = "./geopoi/POIgeo/train"
    rename(path)
    path = "./geopoi/POIgeo/val"
    rename(path)