
import datetime
import io
import matplotlib.pyplot as plt

import numpy as np
import math
from time import time
from datetime import datetime



### to disable GPU
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

import tensorflow as tf
### to disable GPU
from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

### to select GPU
tf.device("gpu:0")



import gpflow

from gpflow.config import default_float
from gpflow.ci_utils import ci_niter
from gpflow.utilities import to_default_float
from gpflow import set_trainable
from gpflow.utilities import print_summary


import warnings
warnings.filterwarnings("ignore")

gpflow.config.set_default_float(np.float32)
np.random.seed(0)
tf.random.set_seed(0)




import rospy
import pcl
import ros_numpy

from sensor_msgs import point_cloud2
from std_msgs.msg import Header
# from gp_msgs.msg  import PCL_obsv, Array_1d
from vsgp_gen_mdl.msg  import PclObsv
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from geometry_msgs.msg import Pose
from scipy import stats



class VSGPScout:
    def __init__(self):
        # Node initialization
        rospy.init_node("vsgp_scout")
        print("Initialize vsgp scout ... ")
        self.smpld_pcl_pub = rospy.Publisher( "/sgp_smpld_pcl", PointCloud2, queue_size=1)
        self.oc_srfc_pcl_pub = rospy.Publisher( "/sgp_oc_srfc", PointCloud2, queue_size=1)
        self.oc_var_pcl_pub = rospy.Publisher( "/sgp_oc_var", PointCloud2, queue_size=1)
        self.sph_pcl_sub = rospy.Subscriber("/lfrq_sph_pcl", PointCloud2, self.sph_pcl_cb) 

        self.x_grad_pcl_pub = rospy.Publisher( "/x_grad", PointCloud2, queue_size=1)
        self.y_grad_pcl_pub = rospy.Publisher( "/y_grad", PointCloud2, queue_size=1)


        self.thetas = None
        self.alphas = None
        self.rds    = None
        self.occs   = None


        self.sgp_model   = None
        self.kernel      = None
        self.kernel1     = None
        self.kernel2     = None
        self.likelihood  = None
        self.mean_func   = None


        self.skip = 3
        self.oc_srfc_rds = 6
        self.oc_srfc_viz_rds = 10
        self.oc_var_viz_rds  = 9
        self.sgp_trng_t = None
        self.pcl_size  = None
        self.induc_pts_size = None
        self.dwnsmpld_pcl_size = None

        # self.header = Header()
        # self.header.seq = None
        # self.header.stamp = None
        # self.header.frame_id =  "velodyne"    #   "camera_init"# "aft_mapped"     #
  
        self.header = Header()
        self.header.seq = 0
        # self.header.stamp = 0
        self.header.frame_id =  "velodyne"    #   "camera_init"# "aft_mapped"     #
        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('intensity', 12, PointField.FLOAT32, 1)]



        self.trng_file = None
        # self.data_files()
        rospy.spin()


    ### path to save data 
    def data_files(self):
        timestr = datetime.now() 
        vsgp_base_dir = "/home/vail/mahmoud_ws/sgp_metrics/vsgp_scout/"
        self.trng_file  = vsgp_base_dir + "scout_" + str(timestr) + ".txt"


    ### save data for further processing
    def save_data(self):
        file = open(self.trng_file, "a")
        np.savetxt( file, np.array( [self.header.seq, self.header.stamp.to_sec(), self.sgp_trng_t, self.pcl_size, self.dwnsmpld_pcl_size, self.induc_pts_size], dtype='float32').reshape(-1,6) , delimiter=" ", fmt='%1.3f')
        file.close()





    def sph_pcl_cb(self, pcl_msg):
        print("\n\n******************** ", pcl_msg.header.seq, pcl_msg.header.stamp.to_sec(), " ********************")
        msg_rcvd_time = time()
      
        ################### extract points from pcl for trainging ###############
        # self.pcl_arr = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pcl_msg, remove_nans=True)
        pcl_arr = ros_numpy.point_cloud2.pointcloud2_to_array(pcl_msg, squeeze = True)       

        self.header.seq = pcl_msg.header.seq
        self.header.stamp = pcl_msg.header.stamp
        self.pcl_size = np.shape(pcl_arr)[0]
        pcl_arr = np.round( np.array(pcl_arr.tolist(), dtype='float'), 4) # np.asarray(self.pcl_arr[0])
        print("pcl_arr shape: ", np.shape(pcl_arr) )



        ### downsample_pcl
        self.downsample_pcl(pcl_arr)
        self.sampling_grid()

        ### limit sensor range 
        # self.limit_sensor_range()


        ### periodicity
        # self.apply_periodicity()


        ### vsgp input datat 
        d_in  = np.column_stack( (self.thetas, self.alphas) ) 
        d_out = np.array(self.occs, dtype='float').reshape(-1,1)
        data_in_tensor = tf.convert_to_tensor(d_in, dtype=tf.float32) 
        data_out_tensor = tf.convert_to_tensor(d_out, dtype=tf.float32) 
        training_data = (data_in_tensor, data_out_tensor)


        ### inducing inputs
        num_inducing_pts = 400 # 500 
        initial_pts = range(0, self.dwnsmpld_pcl_size, int(self.dwnsmpld_pcl_size/num_inducing_pts) )
        inducing_variable = d_in[[r for r in initial_pts], :]
        print("inducing_variables: ", np.shape(inducing_variable) )


        ### select kernel, mean function and likelihood
        self.select_kernel()
        self.select_mean_function()
        self.select_likelihood()


        ### select SGP model
        # self.sgp_model = gpflow.models.SVGP(kernel=kernel, likelihood=likelihood, inducing_variable= inducing_variable)
        self.sgp_model = gpflow.models.SGPR(training_data, self.kernel, inducing_variable, mean_function =self.mean_func)#, noise_variance=0.01)
        # print_summary(sgp_model)

        ### set trainable and non-trainable parameters
        self.set_trainable_param()

        ### optimization 
        self.optimize_param()
        

        ### save data for plotting further processing 
        # self.save_data()
        ### publish sgp_param msgs 
        # self.pub_sgp_param()

        # gp_start_pred_t = time()
        # oc_fun, oc_var = self.sgp_model.predict_f(self.grid)
        # self.gp_pred_t = time()-gp_start_pred_t
        # print("GP prediction time: ", self.gp_pred_t)

        Xtest_tensor = tf.convert_to_tensor(self.grid) #[:, None]
        with tf.GradientTape(
                persistent=True  # this allows us to compute different gradients below
        ) as tape:
            # By default, only Variables are watched. For gradients with respect to tensors,
            # we need to explicitly watch them:
            tape.watch(Xtest_tensor)

            oc_fun, oc_var = self.sgp_model.predict_f(Xtest_tensor)  # or any other predict function

        grad_mean = tape.gradient(oc_fun, Xtest_tensor)
        grad_var = tape.gradient(oc_var, Xtest_tensor)

        self.grad_mean = grad_mean.numpy()
        self.grad_var = grad_var.numpy()

        print("oc_fun: ", oc_fun.shape)
        print("grd_mean: ", self.grad_mean.shape)
        # print("grd_mean: ", self.grad_mean.T.shape)
        # print("grd_mean: ", self.grad_mean.T[0].shape)


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
        self.oc_var_viz()

        self.x_grad_pcl()
        self.y_grad_pcl()

        # self.gradx_srfc_viz()
        # self.grady_srfc_viz()


        print("\n>>total time: ", time() - msg_rcvd_time)




    def sampling_grid(self):
        ### sample uniformaly according to vlp16  azimuth & elevation resolution ###
        th_rsltion = 0.005  # 0.02 # #from -pi to pi rad -> 0 35999
        al_rsltion = 0.005#0.0349  #vpl16 resoltuion is 2 deg (from -15 to 15 deg)
        th_s = np.arange(- np.pi, np.pi-0.002, th_rsltion, dtype='float32')
        al_s = np.arange(np.pi/2-0.261799, np.pi/2+0.261799,   al_rsltion, dtype='float32')
        self.grid = np.array(np.meshgrid(th_s,al_s)).T.reshape(-1,2)
        # print("grid: ", np.shape(self.grid))
   

    def limit_variance(self):
        #### variance threshold 
        var_stats = stats.describe(self.oc_var)
        self.var_mean = var_stats.mean[0]
        self.var_var = var_stats.variance[0]
        self.var_min = var_stats.minmax[0][0]
        self.var_max = var_stats.minmax[1][0]
        print("stats_var: ", self.var_mean, self.var_var, self.var_min, self.var_max)

        var_thrshld = 1.5*self.var_mean - 2*self.var_var # math.sqrt(var_var)
        print("var_thrshld: ", var_thrshld)

        #### limit variance 
        ids = np.where(self.oc_var < var_thrshld)[0]
        # print("self.oc_var > x: ids ", np.shape(ids) )
        self.oc_var = self.oc_var[ids] 
        self.rds_fun = self.rds_fun[ids] 
        self.grid = self.grid[ids][:] 
        self.grad_mean = self.grad_mean[ids] 



    def limit_radius(self):
        ids = np.where(self.rds_fun > 0.3)[0]
        # print("r_mean < x: ids ", np.shape(ids) )
        self.oc_var = self.oc_var[ids] 
        self.rds_fun = self.rds_fun[ids] 
        self.grid = self.grid[ids][:] 

        ids = np.where(self.rds_fun < self.oc_srfc_rds+0.5)[0]
        # print("r_mean < x: ids ", np.shape(ids) )
        self.oc_var = self.oc_var[ids] 
        self.rds_fun = self.rds_fun[ids] 
        self.grid = self.grid[ids][:] 
        self.grad_mean =self.grad_mean[ids] 



    def smpld_pcl(self):
        x, y, z = self.convert_spherical_2_cartesian(self.grid.T[:][0].reshape(-1,1), self.grid.T[:][1].reshape(-1,1), self.rds_fun)
        intensity = np.array( self.oc_var, dtype='float32').reshape(-1, 1)
        smpld_pcl = np.column_stack( (x, y, z, intensity) )
        pc2 = point_cloud2.create_cloud(self.header, self.fields, smpld_pcl)
        self.smpld_pcl_pub.publish(pc2)
    def x_grad_pcl(self):
        x, y, z = self.convert_spherical_2_cartesian(self.grid.T[:][0].reshape(-1,1), self.grid.T[:][1].reshape(-1,1), self.rds_fun)
        intensity = np.array( self.grad_mean.T[0], dtype='float32').reshape(-1, 1)
        # print("shape self.grad_mean[:][0]: ", self.grad_mean.T[0].shape)
        smpld_pcl = np.column_stack( (x, y, z, intensity) )
        pc2 = point_cloud2.create_cloud(self.header, self.fields, smpld_pcl)
        self.x_grad_pcl_pub.publish(pc2)
    def y_grad_pcl(self):
        x, y, z = self.convert_spherical_2_cartesian(self.grid.T[:][0].reshape(-1,1), self.grid.T[:][1].reshape(-1,1), self.rds_fun)
        intensity = np.array( self.grad_mean.T[1], dtype='float32').reshape(-1, 1)
        smpld_pcl = np.column_stack( (x, y, z, intensity) )
        pc2 = point_cloud2.create_cloud(self.header, self.fields, smpld_pcl)
        self.y_grad_pcl_pub.publish(pc2)





    def oc_srfc_viz(self):
        rds = self.oc_srfc_viz_rds*np.ones(np.shape(self.rds_fun)[0], dtype='float32').reshape(-1,1)
        x, y, z = self.convert_spherical_2_cartesian(self.grid.T[:][0].reshape(-1,1), self.grid.T[:][1].reshape(-1,1), rds)
        intensity = np.array( self.rds_fun, dtype='float32').reshape(-1, 1)
        oc_srfc_pcl = np.column_stack( (x, y, z, intensity) )
        pc2 = point_cloud2.create_cloud(self.header, self.fields, oc_srfc_pcl)
        self.oc_srfc_pcl_pub.publish(pc2)
    def gradx_srfc_viz(self):
        rds = self.oc_srfc_viz_rds*np.ones(np.shape(self.rds_fun)[0], dtype='float32').reshape(-1,1)
        x, y, z = self.convert_spherical_2_cartesian(self.grid.T[:][0].reshape(-1,1), self.grid.T[:][1].reshape(-1,1), rds)
        intensity = np.array( self.grad_mean.T[0], dtype='float32').reshape(-1, 1)
        oc_srfc_pcl = np.column_stack( (x, y, z, intensity) )
        pc2 = point_cloud2.create_cloud(self.header, self.fields, oc_srfc_pcl)
        self.x_grad_pcl_pub.publish(pc2)
    def grady_srfc_viz(self):
        rds = self.oc_srfc_viz_rds*np.ones(np.shape(self.rds_fun)[0], dtype='float32').reshape(-1,1)
        x, y, z = self.convert_spherical_2_cartesian(self.grid.T[:][0].reshape(-1,1), self.grid.T[:][1].reshape(-1,1), rds)
        intensity = np.array( self.grad_mean.T[1], dtype='float32').reshape(-1, 1)
        oc_srfc_pcl = np.column_stack( (x, y, z, intensity) )
        pc2 = point_cloud2.create_cloud(self.header, self.fields, oc_srfc_pcl)
        self.y_grad_pcl_pub.publish(pc2)


    def oc_var_viz(self):
        rds = self.oc_var_viz_rds*np.ones(np.shape(self.oc_var)[0], dtype='float32').reshape(-1,1)
        x, y, z = self.convert_spherical_2_cartesian(self.grid.T[:][0].reshape(-1,1), self.grid.T[:][1].reshape(-1,1), rds)
        intensity = np.array( self.oc_var, dtype='float32').reshape(-1, 1)
        oc_var_pcl = np.column_stack( (x, y, z, intensity) )
        pc2 = point_cloud2.create_cloud(self.header, self.fields, oc_var_pcl)
        self.oc_var_pcl_pub.publish(pc2)

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














    def downsample_pcl(self, pcl_arr):  
        ## sort original pcl based on thetas
        pcl_arr = pcl_arr[np.argsort(pcl_arr[:, 0])]
        thetas = pcl_arr.transpose()[:][0].reshape(-1,1)
        unique_thetas = np.array( sorted( set(thetas.flatten())) ) #.reshape(-1,1)
        # print("th_set: ", np.shape(unique_thetas) )

        #### fraction to delete or percentage to keep 
        keep_th_ids = [ t for t in range(0, np.shape(unique_thetas)[0], self.skip)]    
        ids = []
        for t in keep_th_ids:
            ids = ids + list(np.where(thetas == unique_thetas[t] )[0])    

        # self.pcl_arr = np.delete(pcl_arr, ids, 0) # dimension along delete
        pcl_arr = pcl_arr[ids] 
        pcl_arr = pcl_arr.transpose()

        self.thetas = np.round(pcl_arr[:][0].reshape(-1,1), 4 )
        self.alphas = np.round(pcl_arr[:][1].reshape(-1,1), 4 )
        self.rds    = np.round(pcl_arr[:][2].reshape(-1,1), 4 )
        self.occs   = np.round(pcl_arr[:][3].reshape(-1,1), 4 )
        self.dwnsmpld_pcl_size = np.shape(self.thetas)[0]

        print("downsampled size : ", np.shape(self.thetas)[0] )
        unique_smpld_th = np.array( sorted( set(self.thetas.flatten())) ) #.reshape(-1,1)
        unq_smpld_th_size = np.shape(unique_smpld_th)[0]
        print("unique_smpld_th: ", unq_smpld_th_size )



    def select_mean_function(self):
        self.mean_func = gpflow.mean_functions.Constant(0) 



    def select_likelihood(self):
        # likelihood = gpflow.likelihoods.Exponential()
        # model = gpflow.models.GPMC(data, kernel, likelihood)
        self.likelihood = gpflow.likelihoods.Gaussian()
        self.likelihood.variance.assign(.01) #0.01




    def select_kernel(self):
        self.kernel1 = gpflow.kernels.RationalQuadratic(lengthscales=[0.09, 0.11])#[0.12, 0.14])
        self.kernel2= gpflow.kernels.White(10)
        self.kernel2.variance.assign(5e-3) #0.005
        self.kernel1.variance.assign(.7) #0.7
        # self.kernel1.lengthscales.assign(.3) #0.4
        self.kernel1.alpha.assign(10) #10
        self.kernel = self.kernel1 + self.kernel2



    def set_trainable_param(self):
        set_trainable(self.kernel1.variance, False)
        set_trainable(self.kernel1.lengthscales, False)
        # set_trainable(self.kernel1.alpha, False)
        set_trainable(self.kernel2.variance, False)
        set_trainable(self.sgp_model.likelihood.variance, False)
        # set_trainable(self.sgp_model.inducing_variable, False)



    def optimize_param(self):
        start_training = time()
        optimizer = tf.optimizers.Adam()
        optimizer.minimize(self.sgp_model.training_loss, self.sgp_model.trainable_variables )  
        self.sgp_trng_t = time()-start_training
        print("self.sgp_trng_t: ", self.sgp_trng_t)



       

    def pub_sgp_param(self):
        ### inducing variables
        inducing_pts = self.sgp_model.inducing_variable.Z.numpy()
        induc_pts_occ, var = self.sgp_model.predict_f(inducing_pts)
        induc_pts_occ = induc_pts_occ.numpy().reshape(-1, 1)
        self.induc_pts_size = np.shape(inducing_pts)[0]

        inducing_pts_data = np.column_stack( (inducing_pts, induc_pts_occ) )

        print("self.induc_pts_size: ", self.induc_pts_size )
        print("inducing_pts_data: ", np.shape(inducing_pts_data) )

        
        ## create sgp_param msg and send it to base 
        sgp_msg = PclObsv()
        sgp_msg.header = self.header
        sgp_msg.kernel_param.append( self.kernel1.variance.numpy() )
        sgp_msg.kernel_param.append( self.kernel1.lengthscales.numpy()[0] )
        sgp_msg.kernel_param.append( self.kernel1.lengthscales.numpy()[1] )
        sgp_msg.kernel_param.append( self.kernel1.alpha.numpy() )
        sgp_msg.likelihood_var = self.likelihood.variance.numpy()
        sgp_msg.noise_var      = self.kernel2.variance.numpy()

        for pt in inducing_pts_data:
            sgp_msg.inducing_points.append(pt[0])
            sgp_msg.inducing_points.append(pt[1])
            sgp_msg.inducing_points.append(pt[2])
        self.sgp_param_pub.publish(sgp_msg)








    def apply_periodicity(self):
        ############################## Peroidicity ############################## 
        ###### duplicate data set
        # thetas_aug = np.append( thetas, 2*np.pi+thetas).reshape(-1,1)
        # alphas_aug = np.append( alphas, alphas).reshape(-1,1)
        # rds_aug = np.append( rds, rds).reshape(-1,1)
        # occs_aug = np.append( occs, occs).reshape(-1,1)

        # thetas_aug = np.append( thetas_aug, -2*np.pi+thetas).reshape(-1,1)
        # alphas_aug = np.append( alphas_aug, alphas).reshape(-1,1)
        # rds_aug = np.append( rds_aug, rds).reshape(-1,1)
        # occs_aug = np.append( occs_aug, occs).reshape(-1,1)



        ## max alpha value = 1.83259535
        ## min alpha value = 1.30899727

        ##### add portion of data to represent the peroidicity 
        ## 512: should be flixable based on number of points you receive
        ### should be function of the 

        idx = max_N_elements_idx(thetas.reshape(-1), int(unq_smpld_th_size/10)) 
        thetas_aug = np.append(thetas, -2*np.pi + thetas[idx] ).reshape(-1,1)
        alphas_aug = np.append(alphas, alphas[idx]).reshape(-1,1)
        rds_aug = np.append(rds, rds[idx]).reshape(-1,1)
        occs_aug = np.append(occs, occs[idx]).reshape(-1,1)

        idx = min_N_elements_idx(thetas.reshape(-1), int(unq_smpld_th_size/10)) 
        thetas_aug = np.append(thetas_aug, 2*np.pi + thetas[idx] ).reshape(-1,1)
        alphas_aug = np.append(alphas_aug, alphas[idx]).reshape(-1,1)
        rds_aug = np.append(rds_aug, rds[idx]).reshape(-1,1)
        occs_aug = np.append(occs_aug, occs[idx]).reshape(-1,1)

        ##3600: should be flixable based on number of points you receive
        idx = min_N_elements_idx(alphas.reshape(-1), unq_smpld_th_size)  ##should be flixable based on number of points you receive
        alphas_aug = np.append(alphas_aug, alphas[idx] - 0.01).reshape(-1,1)
        thetas_aug = np.append(thetas_aug, thetas[idx]).reshape(-1,1)
        rds_aug = np.append(rds_aug, rds[idx]).reshape(-1,1)
        occs_aug = np.append(occs_aug, occs[idx]).reshape(-1,1)

        idx = max_N_elements_idx(alphas.reshape(-1), unq_smpld_th_size) 
        alphas_aug = np.append(alphas_aug, alphas[idx] + 0.01).reshape(-1,1)
        thetas_aug = np.append(thetas_aug, thetas[idx]).reshape(-1,1)
        rds_aug = np.append(rds_aug, rds[idx]).reshape(-1,1)
        occs_aug = np.append(occs_aug, occs[idx]).reshape(-1,1)




    def limit_sensor_range(self):
        #### limit sensor range to 5  
        ids = np.where(self.occs < 8.0)[0]   #### replaced by "r_mean[r_mean < 0.4] = 0.4"
        print("occs < x: ids ", np.shape(ids) )
        self.thetas = self.thetas[ids] 
        self.alphas = self.alphas[ids] 
        self.rds = self.rds[ids] 
        self.occs = self.occs[ids] 




    def min_N_elements_idx(self, arr, N):
        return arr.argsort()[:N]

    def max_N_elements_idx(self, arr, N):
        return arr.argsort()[-N:]





if __name__ =='__main__':
    try:
        VSGPScout()
    except rospy.ROSInterruptException:
        pass
















