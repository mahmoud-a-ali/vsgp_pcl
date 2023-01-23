## !/usr/bin/env python
import rospy
import yaml
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Bool



# dir_to_save = "/home/robotics/Mahmoud_ws/github/way_points/cave_track/"
dir_to_save = "/home/vail/Mahmoud_ws/vsgp_pcl_ws/src/waypts_nav_pid/cave_track/"


# waypts_pub = rospy.Publisher("/waypoint", PointStamped, queue_size = 10)
move_goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size = 1)


glb_wpt = 1

def waypt_status_cb( status_msg ):
    global glb_wpt
    num_wpts = 100 +1
 
    
    if (status_msg.data == True ) & (glb_wpt < num_wpts):
        print(">>>>>>>>>> send waypoint: ", glb_wpt)
        # glb_wpt += 1
        # if glb_wpt%4 == 1:
        #     return 0
     


        file = dir_to_save + "pose" + str(glb_wpt) + ".yaml"
        with open(file, 'r') as stream:
            try:
                prsd_pose=yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        gl_msg = PoseStamped()
        gl_msg.header.seq = glb_wpt
        gl_msg.header.stamp = rospy.get_rostime() #rospy.get_time()
        # gl_msg.header.frame_id = 'world'
        gl_msg.header.frame_id = 'camera_init'
        gl_msg.pose.position.x =  prsd_pose['position']['x']
        gl_msg.pose.position.y =  prsd_pose['position']['y']
        gl_msg.pose.position.z =  prsd_pose['position']['z']
        gl_msg.pose.orientation.x =  prsd_pose['orientation']['x']
        gl_msg.pose.orientation.y =  prsd_pose['orientation']['y']
        gl_msg.pose.orientation.z =  prsd_pose['orientation']['z']
        gl_msg.pose.orientation.w =  prsd_pose['orientation']['w']
        move_goal_pub.publish(gl_msg)

        # wypt_msg = PointStamped()
        # wypt_msg.header.seq = glb_wpt
        # wypt_msg.header.stamp = rospy.get_rostime() #rospy.get_time()
        # wypt_msg.header.frame_id = 'world'
        # wypt_msg.point.x =  prsd_pose['position']['x']
        # wypt_msg.point.y =  prsd_pose['position']['y']

        ### get eurler angel for z aas it is yaw angel
        gl_msg.pose.position.z =  prsd_pose['position']['z']
        glb_wpt += 1

        print(gl_msg)
        # rospy.sleep(1)



def waypt_nav():
    print("waypt_nav ... ")
    rospy.init_node('send_waypt', anonymous=False)
    rospy.Subscriber("/need_waypoint", Bool, waypt_status_cb, queue_size=10 )
    
    rospy.spin()
 

        # spin() simply keeps python from exiting until this node is stopped
        # rospy.spin()






if __name__ == '__main__':
    waypt_nav()