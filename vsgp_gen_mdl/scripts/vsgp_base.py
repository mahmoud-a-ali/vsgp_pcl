from typing import Tuple, Optional
import tempfile
import pathlib
import warnings


import io
import os
import math
import numpy as np
from time import time
from scipy import stats
from datetime import datetime


import pcl
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
from vsgp_gen_mdl.msg import PclObsv
from geometry_msgs.msg import Pose


from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt


### to disable GPU: (working)
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf


### to disable GPU: another option (not working)
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

### to select GPU
# tf.device("gpu:0")


## after tensorflow import gpflow
import gpflow
from gpflow.config import default_float
from gpflow.ci_utils import ci_niter
from gpflow.utilities import to_default_float
from gpflow import set_trainable
from gpflow.utilities import print_summary


#### configurations
warnings.filterwarnings("ignore")
gpflow.config.set_default_float(np.float32)
np.random.seed(0)
tf.random.set_seed(0)



## streaming sparse not available yet, check 
## https://gpflow.readthedocs.io/en/master/notebooks/tailor/updating_models_with_new_data.html

## they use VFE for sparse approximation, check 
## https://gpflow.readthedocs.io/en/master/notebooks/theory/FITCvsVFE.html

## equations for SGPR/VSGP based on Titas2009, check 
## https://gpflow.readthedocs.io/en/master/notebooks/theory/SGPR_notes.html

## different types of GP in GPflow, check 
## https://gpflow.readthedocs.io/en/master/notebooks/theory/Sanity_check.html



class VSGPBase:
    def __init__(self):
        # Node initialization
        rospy.init_node("vsgp_base")
        print("Initialize vsgp base ... ")
        self.sgp_param_sub = rospy.Subscriber("/sgp_param", PclObsv, self.sgp_callback)
        self.smpld_pcl_pub = rospy.Publisher( "/sgp_smpld_pcl", PointCloud2, queue_size=1)
        self.oc_srfc_pcl_pub = rospy.Publisher( "/sgp_oc_srfc", PointCloud2, queue_size=1)
        self.oc_var_pcl_pub = rospy.Publisher( "/sgp_oc_var", PointCloud2, queue_size=1)


        self.kernel_param   = None
        self.noise_var      = None
        self.likelihood_var = None


        self.oc_srfc_rds     = 8
        self.oc_srfc_viz_rds = 10
        self.oc_var_viz_rds  = 12

        self.grid = None
        self.oc_fun = None
        self.oc_var = None
        self.rds_fun = None

        self.var_mean = None
        self.var_var  = None
        self.var_min  = None
        self.var_max  = None

        self.gp_trng_t = 0.0
        self.gp_pred_t = 0.0


        self.header = Header()
        self.header.seq = 0
        # self.header.stamp = 0
        self.header.frame_id =  "velodyne"    #   "camera_init"# "aft_mapped"     #
        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('intensity', 12, PointField.FLOAT32, 1)]


        self.param_file = None
        self.trng_pred_var_file = None

        # self.data_files()
        rospy.spin()


    def data_files(self):
        timestr = datetime.now() 
        vsgp_base_dir = "/home/alimaa/github/sgp_metrics/vsgp_base/"
        itr_folder = vsgp_base_dir + "base_" + str(timestr)
        if not os.path.exists(itr_folder):
            os.mkdir(itr_folder)

        self.param_file  = itr_folder +"/hyper_param.txt"
        self.trng_pred_var_file = itr_folder +"/trng_pred_var.txt"




    def sgp_callback(self, sgp_param_msg):
        print("\n\n******************** ", sgp_param_msg.header.seq, sgp_param_msg.header.stamp.to_sec(), " ********************")
        msg_rcvd_time = time()
        #################### extract points from pcl for trainging ####################    
        # print("sgp_param_msg: ", sgp_param_msg)
        self.header.seq = sgp_param_msg.header.seq
        self.header.stamp= sgp_param_msg.header.stamp#.to_sec()

        ########## extract data #########
        data = np.array( sgp_param_msg.inducing_points, dtype='float32').reshape(-1,3)
        robot_pose = sgp_param_msg.pose
        self.kernel_param = sgp_param_msg.kernel_param
        self.noise_var = sgp_param_msg.noise_var
        self.likelihood_var = sgp_param_msg.likelihood_var

        thetas = np.array(data.transpose()[:][0], dtype='float32').reshape(-1,1)
        alphas = np.array(data.transpose()[:][1], dtype='float32').reshape(-1,1)
        radius = np.array(data.transpose()[:][2], dtype='float32').reshape(-1,1)
        ind_pts_num = np.shape(thetas)[0]

        kernel1 = gpflow.kernels.RationalQuadratic(lengthscales= [self.kernel_param[1], self.kernel_param[2]])
        kernel2 = gpflow.kernels.White(10)
        kernel = kernel1 + kernel2
        meanf = gpflow.mean_functions.Constant() #1.0, 0.0) mean and variance I think 
        likelihood = gpflow.likelihoods.Gaussian()


        data_in_tensorc = tf.convert_to_tensor( np.column_stack( (thetas, alphas) ), dtype=tf.float32) 
        data_out_tensorc = tf.convert_to_tensor(radius, dtype=tf.float32) 
        datac = (data_in_tensorc, data_out_tensorc)  

        gp = gpflow.models.GPR(datac, kernel)#, ZZ_tnsr) #[250:,:])
 
        #########  trainable or not #########  
        set_trainable(kernel2.variance, False)
        set_trainable(kernel1.variance, False)
        set_trainable(kernel1.lengthscales, False)
        # set_trainable(kernel1.alpha, False)
        set_trainable(gp.likelihood.variance, False)
        # set_trainable(gp.inducing_variable, False)
        
        ######### assign values as fixed or initial #########
        kernel2.variance.assign(self.noise_var) #0.005
        kernel1.variance.assign(self.kernel_param[0]) #0.7
        # kernel1.lengthscales.assign(.3) #0.4
        kernel1.alpha.assign(self.kernel_param[3]) #10
        gp.likelihood.variance.assign(self.likelihood_var) #0.01

   
        ########################### optimization  ###########################
        gp_strt_trng_t = time()
        optimizer = tf.optimizers.Adam()
        optimizer.minimize(gp.training_loss, gp.trainable_variables )  
        self.gp_trng_t = time()- gp_strt_trng_t
        print("GP training time: ", self.gp_trng_t)


        ##################### sampling #####################
        strt_smplng_t = time()
        self.sampling_grid()


        ###### prediction
        gp_start_pred_t = time()
        oc_fun, oc_var = gp.predict_f(self.grid)
        self.gp_pred_t = time()-gp_start_pred_t
        print("GP prediction time: ", self.gp_pred_t)

        ##### occupancy to spherical radius
        self.oc_var = oc_var.numpy()
        self.rds_fun = self.oc_srfc_rds - oc_fun.numpy()
        # print("self.rds_fun: ", np.shape(self.rds_fun))
        
        #### limit variance 
        self.limit_variance()


        #### limit radius 
        self.limit_radius()
        # print("shape of grid: ", np.shape(self.grid) )
        # print("shape of mean: ", np.shape(self.rds_fun) )

        # self.save_data()
        self.smpld_pcl()
        self.oc_srfc_viz()
        # self.oc_var_viz()
        print("time for prediction & sampling: ", time()- strt_smplng_t)




    def limit_variance(self):
        #### variance threshold 
        var_stats = stats.describe(self.oc_var)
        self.var_mean = var_stats.mean[0]
        self.var_var = var_stats.variance[0]
        self.var_min = var_stats.minmax[0][0]
        self.var_max = var_stats.minmax[1][0]
        print("stats_var: ", self.var_mean, self.var_var, self.var_min, self.var_max)

        var_thrshld = self.var_mean - 2*self.var_var # math.sqrt(var_var)
        print("var_thrshld: ", var_thrshld)

        #### limit variance 
        ids = np.where(self.oc_var < var_thrshld)[0]
        # print("self.oc_var > x: ids ", np.shape(ids) )
        self.oc_var = self.oc_var[ids] 
        self.rds_fun = self.rds_fun[ids] 
        self.grid = self.grid[ids][:] 




    def limit_radius(self):
        ids = np.where(self.rds_fun > 0.4)[0]
        # print("r_mean < x: ids ", np.shape(ids) )
        self.oc_var = self.oc_var[ids] 
        self.rds_fun = self.rds_fun[ids] 
        self.grid = self.grid[ids][:] 

        ids = np.where(self.rds_fun < 8.0)[0]
        # print("r_mean < x: ids ", np.shape(ids) )
        self.oc_var = self.oc_var[ids] 
        self.rds_fun = self.rds_fun[ids] 
        self.grid = self.grid[ids][:] 




    def sampling_grid(self):
        ### sample uniformaly according to vlp16  azimuth & elevation resolution ###
        th_rsltion = 0.00174  # 0.02 # #from -pi to pi rad -> 0 35999
        al_rsltion = 0.01#0.0349  #vpl16 resoltuion is 2 deg (from -15 to 15 deg)
        th_s = np.arange(- np.pi, np.pi-0.002, th_rsltion, dtype='float32')
        al_s = np.arange(np.pi/2-0.261799, np.pi/2+0.261799,   al_rsltion, dtype='float32')
        self.grid = np.array(np.meshgrid(th_s,al_s)).T.reshape(-1,2)
        # print("grid: ", np.shape(self.grid))



    def convert_spherical_2_cartesian(self, theta, alpha, dist):
        x = np.array( dist * np.sin(alpha) * np.cos(theta), dtype='float32').reshape(-1,1)
        y = np.array( dist * np.sin(alpha) * np.sin(theta), dtype='float32').reshape(-1,1)
        z = np.array( dist * np.cos(alpha) , dtype='float32').reshape(-1,1)
        return x, y, z


    def convert_cartesian_2_spherical(self, x, y, z):
        dist = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        alpha = np.arccos(z / dist)
        return theta, alpha, dist



    def save_data(self):
        ## save paramters
        file = open(self.param_file, "a")
        np.savetxt( file, np.array( [self.header.seq, self.header.stamp.to_sec(), self.kernel_param[0], self.kernel_param[1], 
                                    self.kernel_param[2], self.kernel_param[3], self.noise_var, self.likelihood_var, self.oc_srfc_rds], 
                                    dtype='float32').reshape(-1,9) , delimiter=" ", fmt='%1.4f')
        file.close()
        ## save timing (training & prediction)
        file = open(self.trng_pred_var_file, "a")
        np.savetxt( file, np.array( [self.header.seq, self.header.stamp.to_sec(), self.gp_trng_t, self.gp_pred_t, self.var_mean, self.var_var, self.var_min, self.var_max], 
                                    dtype='float32').reshape(-1,8) , delimiter=" ", fmt='%1.4f')
        file.close()




    def smpld_pcl(self):
        x, y, z = self.convert_spherical_2_cartesian(self.grid.T[:][0].reshape(-1,1), self.grid.T[:][1].reshape(-1,1), self.rds_fun)
        intensity = np.array( self.oc_var, dtype='float32').reshape(-1, 1)
        smpld_pcl = np.column_stack( (x, y, z, intensity) )
        pc2 = point_cloud2.create_cloud(self.header, self.fields, smpld_pcl)
        self.smpld_pcl_pub.publish(pc2)


    def oc_srfc_viz(self):
        rds = self.oc_srfc_viz_rds*np.ones(np.shape(self.rds_fun)[0], dtype='float32').reshape(-1,1)
        x, y, z = self.convert_spherical_2_cartesian(self.grid.T[:][0].reshape(-1,1), self.grid.T[:][1].reshape(-1,1), rds)
        intensity = np.array( self.rds_fun, dtype='float32').reshape(-1, 1)
        oc_srfc_pcl = np.column_stack( (x, y, z, intensity) )
        pc2 = point_cloud2.create_cloud(self.header, self.fields, oc_srfc_pcl)
        self.oc_srfc_pcl_pub.publish(pc2)


    def oc_var_viz(self):
        rds = self.oc_var_viz_rds*np.ones(np.shape(self.oc_var)[0], dtype='float32').reshape(-1,1)
        x, y, z = self.convert_spherical_2_cartesian(self.grid.T[:][0].reshape(-1,1), self.grid.T[:][1].reshape(-1,1), rds)
        intensity = np.array( self.oc_var, dtype='float32').reshape(-1, 1)
        oc_var_pcl = np.column_stack( (x, y, z, intensity) )
        pc2 = point_cloud2.create_cloud(self.header, self.fields, oc_var_pcl)
        self.oc_var_pcl_pub.publish(pc2)






if __name__ == "__main__":
    try:
        VSGPBase()
    except rospy.ROSInterruptException:
        pass















