import os
from cv2 import cv
from multiviewtamin.reconstruct import triangulate_dlt
from multiviewtamin.camera import load_cams_yaml
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

# import camera positions
cams = load_cams_yaml('/home/wolle/multiviewtamin/patterncalib2_calib_full.yml')
# directory with yml files
rootdir = '/home/wolle/multiviewtamin/pose/'

data = []
for subdir, dirs, files in os.walk(rootdir):
    files.sort()
    frame = []
    i=0
    for file in files:
        if file.endswith((".yml")):
            absfilepath=os.path.join(subdir, file)
            frame.append(np.array(cv.Load(absfilepath, cv.CreateMemStorage(), "pose_0")))
            # remove the first column from frame dataset (number of humans)
            frame[i] = np.delete(frame[i][0, :, :, ], 2, axis=1)
            # remove rows with points at x,y,z=0
            # column[i] = column[i][~np.all(column[i] == 0, axis=1)]

            i+=1
    # check if empty
    if frame:
        data.append(frame)

# We need to get some Data:
# Number of Frames
numFrames=len(data[0])
# Number of Points per Frame
numPts=len(data[0][0])
# Number of Cameras
numCams=len(data)

# Triangulate
triangP= []
pts2d=[]
for i in xrange(numFrames):
    triangP.append([])

# iterate over points in frames
for l in range(numPts):
    p=0
    # iterate over frames:
    for k in range(len(data[p])):

        # iterate over cameras and collect points
        for i in range (len(data)):
            # Making sure the point isn't missing
            # if not l >= len(data[i][k]):
                # collect the 2d points
            pts2d.append(data[i][k][l])
            p += 1
        triangP[k].append(triangulate_dlt([cam.P_world_to_view for cam in cams], (pts2d)))
        # triangP.append(triangulate_dlt([cam.P_world_to_view for cam in cams], (pts2d)))
        pts2d = []

# Convert to Triangulated Points List to Array for easier handling
triangP_arr = np.asarray(triangP)

# Do 3d animation with matplot: https://stackoverflow.com/a/40896192

# Dimensions
DIM = 3

# flatten array
x = np.concatenate(triangP_arr[:,:,0]).ravel()
y = np.concatenate(triangP_arr[:,:,1]).ravel()
z = np.concatenate(triangP_arr[:,:,2]).ravel()

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

# Setting the axes properties
border = 2000
ax.set_xlim3d([0, border])
ax.set_ylim3d([0, border])
ax.set_zlim3d([-border, border])


def animate(i):
    global x, y, z, numPts
    ax.clear()
    ax.set_xlim3d([0, border])
    ax.set_ylim3d([0, border])
    ax.set_zlim3d([-border, border])
    idx0 = i * numPts
    idx1 = numPts * (i + 1)
    ax.scatter(x[idx0:idx1],y[idx0:idx1],z[idx0:idx1])

ani = animation.FuncAnimation(fig, animate, frames=numFrames, interval=1, blit=False, repeat=False)
#save outout as .gif
ani.save('3d_plot.gif',dpi=50, writer='imagemagick', fps=31)
#plt.show()