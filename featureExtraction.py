#!/usr/bin/env python
# imports

import numpy as np                     # numeric python lib
#
import matplotlib.image as mpimg       # reading images to numpy arrays
import matplotlib.pyplot as plt        # to plot any graph
import matplotlib.patches as mpatches  # to draw a circle at the mean contour


from skimage import measure            # to find shape contour
import scipy.ndimage as ndi            # to determine shape centrality



#
#
# # matplotlib setup
#%matplotlib inline

from pylab import rcParams

def cart2pol(x,y):
    r = np.sqrt(x*x + y*y)
    phi = np.arctan2(y,x)
    return[r, phi]



rcParams['figure.figsize'] = (6, 6)      # setting default size of plots
img = mpimg.imread('./inputs/images/53.jpg')
#img = mpimg.imread('./test.png')


cy, cx = ndi.center_of_mass(img)


print ("cy: " + str(cy) + "\ncx: "+ str(cx))

print(str(np.max(img))   + " " + str(np.min(img)))

contours = measure.find_contours(img, .8)
#

contour = max(contours, key=len)

contour[::,0] -= cy
contour[::,1] -= cx


polar_contour = np.array([cart2pol(x,y) for x, y in contour])

polar_contour  = polar_contour[polar_contour[:,1].argsort(), :] 

plt.scatter(polar_contour[::,1], polar_contour[::,0], linewidth=0, s=1)
plt.show()

#plt.plot(-contour[::, 1], -contour[::,0], linewidth=0.5)
#plt.show()

#plt.imshow(img, cmap='Set3')  # show me the leaf
#plt.scatter(cx, cy)           # show me its center
#plt.show()

