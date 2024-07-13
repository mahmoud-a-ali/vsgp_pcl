
#include <iostream>
#include <ros/ros.h>
#include <vector>
#include <math.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h> //for sensor_msgs::PointCloud2
#include <geometry_msgs/TransformStamped.h>

#include <pcl/point_types.h> // for pcl::PointXYZ
#include <pcl_conversions/pcl_conversions.h> // for pcl::fromROSMsg()

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
// #include <pcl_ros/transforms.hpp>


#include <pcl/segmentation/sac_segmentation.h> //for SACSegmentation
#include <pcl/filters/extract_indices.h> //for ExtractIndices
#include <pcl/filters/statistical_outlier_removal.h>

#include <tf2_ros/transform_listener.h>
#include <tf/transform_listener.h>
#include<tf/tf.h>



#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>


#include <Eigen/Dense>

ros::Publisher lfrq_org_pcl_pub;
ros::Publisher lfrq_sph_pcl_pub;
ros::Publisher lfrq_org_oc_srfc_pub;

// with PointCloud2Ptr not working, but with PointCloud2::ConstPtr it works 
// void sync_callback(const boost::shared_ptr<const sensor_msgs::PointCloud2>& pcl_in, const nav_msgs::Odometry::ConstPtr &pose_in)



float px_old , py_old, yaw_old;



int sync_callback(const  sensor_msgs::PointCloud2::ConstPtr &pcl_in, const nav_msgs::Odometry::ConstPtr &pose_in)
{
  // Solve all of perception here...
  ROS_INFO("\n\nrcvd msg ... ");

  ROS_INFO_STREAM("pcl  time stamp: "<< pcl_in->header.stamp);
  ROS_INFO_STREAM("pose time stamp: "<< pose_in->header.stamp); 



  //*********** Key frames***********
  tf::Quaternion q( pose_in->pose.pose.orientation.x, pose_in->pose.pose.orientation.y, 
                    pose_in->pose.pose.orientation.z, pose_in->pose.pose.orientation.w );

  tf::Matrix3x3 m(q);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);

  // if( abs(yaw - yaw_old) > 0.2  ){
  //   print(" larger:")
  //   return 0;
  // }
  
  // if( abs(pose_in->pose.pose.position.x - px_old)<0.7 && abs(pose_in->pose.pose.position.y - py_old) < 0.7  )
    // return 0;
  


  px_old = pose_in->pose.pose.position.x;
  py_old = pose_in->pose.pose.position.y; 
  yaw_old = yaw;






  //convert the input into a pcl::PointCloud< pcl::PointXYZ> object
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);


  // variables  for filtered/transformed/low_frq  pcl 
  pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloudPtr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*pcl_in, *cloudPtr);



  //*********** pcl ***********
  pcl::PointCloud<pcl::PointXYZI> org_pcl, sph_pcl, occ_srfc_pcl;
  std::copy(cloudPtr->points.begin(), cloudPtr->points.end(),std::back_inserter(org_pcl));



  //*********** clustering ***********
  // next step



 
  //*********** project to different form ***********
  std::vector<pcl::PointXYZI> sph_pcl_vec, occ_srfc_pcl_vec;
  for(auto pt: org_pcl){
    // ROS_INFO_STREAM("pt: " << pt.x);
    float dst = sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z );
    float th = atan2(pt.y , pt.x );
    float al = acos(pt.z / dst );

    // spherical form
    float viz_oc_srfc_rds = 8; // occupancy surface radius
    pcl::PointXYZI shp_pt;
    shp_pt.x = th;
    shp_pt.y = al;
    shp_pt.z = dst;
    shp_pt.intensity = 8 - dst;
    sph_pcl_vec.push_back(shp_pt);

   
    // occupancy surface 
    pcl::PointXYZI oc_srfc_pt;
    oc_srfc_pt.x = viz_oc_srfc_rds * sin(al) * cos(th);
    oc_srfc_pt.y = viz_oc_srfc_rds * sin(al) * sin(th);
    oc_srfc_pt.z = viz_oc_srfc_rds * cos(al); 
    oc_srfc_pt.intensity = dst;
    occ_srfc_pcl_vec.push_back(oc_srfc_pt);
  }
  ROS_INFO_STREAM("sph_pcl_vec: "<< sph_pcl_vec.size() );
  ROS_INFO_STREAM("occ_srfc_pcl_vec: "<< occ_srfc_pcl_vec.size() );




  //*********** std vector to pcl ***********
  std::copy(sph_pcl_vec.begin(), sph_pcl_vec.end(),std::back_inserter(sph_pcl));
  sph_pcl.header.frame_id = "velodyne"; 

  std::copy(occ_srfc_pcl_vec.begin(), occ_srfc_pcl_vec.end(),std::back_inserter(occ_srfc_pcl));
  occ_srfc_pcl.header.frame_id = "velodyne"; 

  ROS_INFO_STREAM("sph_pcl: "<< sph_pcl.size());
  ROS_INFO_STREAM("occ_srfc_pcl: "<< occ_srfc_pcl.size());


  //*********** from velodyne to world frame ***********
  // Eigen::Quaternionf rotation(pose_in->pose.pose.orientation.w, pose_in->pose.pose.orientation.x, pose_in->pose.pose.orientation.y,
                              // pose_in->pose.pose.orientation.z);    
  // Eigen::Vector3f origin(pose_in->pose.pose.position.x, pose_in->pose.pose.position.y, pose_in->pose.pose.position.z+0.322);
  

  // transformPointCloud(org_pcl, *transformed_cloudPtr, origin, rotation);
  // transformed_cloudPtr->header.frame_id = "world"; //should be after transform otherwise aha
  // ROS_INFO_STREAM("transformed_cloudPtr: "<< transformed_cloudPtr->points.size());



//*********** Publish msgs ***********
  sensor_msgs::PointCloud2 sph_pcl_msg;
  pcl::toROSMsg(sph_pcl, sph_pcl_msg);
  sph_pcl_msg.header.stamp = pcl_in->header.stamp; //.toSec(); 
  lfrq_sph_pcl_pub.publish(sph_pcl_msg);


  sensor_msgs::PointCloud2 occ_srfc_pcl_msg;
  pcl::toROSMsg(occ_srfc_pcl, occ_srfc_pcl_msg);
  sph_pcl_msg.header.stamp = pcl_in->header.stamp;//.toSec(); 
  lfrq_org_oc_srfc_pub.publish(occ_srfc_pcl_msg);


  lfrq_org_pcl_pub.publish(pcl_in);
  // ros::Duration(0.3).sleep();

  return 0;
}




int main(int argc, char** argv)
{
  ros::init(argc, argv, "pcl_surface_projection");
  ROS_INFO("pcl_surface_projection node ... ");

  ros::NodeHandle nh;

  //create publisher object
  lfrq_sph_pcl_pub = nh.advertise<sensor_msgs::PointCloud2>("/lfrq_sph_pcl", 1);
  lfrq_org_pcl_pub = nh.advertise<sensor_msgs::PointCloud2>("/lfrq_org_pcl", 1);
  lfrq_org_oc_srfc_pub = nh.advertise<sensor_msgs::PointCloud2>("/lfrq_org_oc_srfc", 1);

  // message filter time synchronization
  message_filters::Subscriber<nav_msgs::Odometry> pose_sub(nh, "/ground_truth/state", 1);
  message_filters::Subscriber<sensor_msgs::PointCloud2> pcl_sub(nh, "/mid/points", 1);
  message_filters::TimeSynchronizer<sensor_msgs::PointCloud2, nav_msgs::Odometry> sync(pcl_sub, pose_sub, 10);
 
  sync.registerCallback(boost::bind(&sync_callback, _1, _2)); // try std instead boost not working as well
  ros::spin();

  return 0;
}


