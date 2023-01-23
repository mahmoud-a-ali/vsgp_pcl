## !/usr/bin/env python
import rospy
import yaml
import numpy as np
import argparse
from scipy import stats
import math

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib.font_manager as font_manager
# It's also possible to use the reduced notation by directly setting font.family:
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "Helvetica",
  'font.size': 12,

})



dir_to_save = "/home/robotics/Mahmoud_ws/vsgp_pcl_ws/src/waypts_nav_pid/cave_track/"


num_wpts = 300
Px = []
Py = []

for pt in range(1, num_wpts):
    if pt in [ p for p in range(301, 302)] :
        continue



    file = dir_to_save + "pose_" + str(pt) + ".yaml"
    with open(file, 'r') as stream:
        try:
            prsd_pose=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

        Px.append( prsd_pose['position']['x'])
        Py.append( prsd_pose['position']['y'])


        # px =  prsd_pose['position']['x']
        # py =  prsd_pose['position']['y']
        # pz =  prsd_pose['position']['z']
        # qx =  prsd_pose['orientation']['x']
        # qy =  prsd_pose['orientation']['y']
        # qz =  prsd_pose['orientation']['z']
        # qw =  prsd_pose['orientation']['w']



Px_arr = np.array(Px, dtype='float32')
Py_arr = np.array(Py, dtype='float32')

dx_arr = np.diff(Px_arr, axis=0)
dy_arr = np.diff(Py_arr, axis=0)



dist_arr = np.sqrt(dx_arr**2+dy_arr**2)
dist = np.sum(dist_arr)

# print("dx_arr: ", dx_arr)
# print("dy_arr: ", dy_arr)
# print("dist_arr: ", dist_arr)
print("dist: ", dist)

#################### plot xy path  ####################  
fig, ax0 = plt.subplots(nrows=1, sharex=True, figsize=(8,3))

ax0.plot( Px, Py, '*')

ax0.set_xlabel('x ')
ax0.set_ylabel('y')
# ax0.legend()

# plt.savefig(fig_dir + '/jackal_Cave_path.pdf', bbox_inches = 'tight', pad_inches = 0)
plt.show()



