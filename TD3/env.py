import rospy
import subprocess
from os import path
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import numpy as np
import math
from gazebo_msgs.msg import ModelState
from squaternion import Quaternion
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(0)

# 检查生成的目标点是否在障碍物上, 随机生成目标点
def check_pos(x, y):
    goalOK = True
    if -3.8 > x > -6.2 and 6.2 > y > 3.8: goalOK = False
    if -1.3 > x > -2.7 and 4.7 > y > -0.2: goalOK = False
    if -0.3 > x > -4.2 and 2.7 > y > 1.3: goalOK = False
    if -0.8 > x > -4.2 and -2.3 > y > -4.2: goalOK = False
    if -1.3 > x > -3.7 and -0.8 > y > -2.7: goalOK = False
    if 4.2 > x > 0.8 and -1.8 > y > -3.2: goalOK = False
    if 4 > x > 2.5 and 0.7 > y > -3.2: goalOK = False
    if 6.2 > x > 3.8 and -3.3 > y > -4.2: goalOK = False
    if 4.2 > x > 1.3 and 3.7 > y > 1.5: goalOK = False
    if -3.0 > x > -7.2 and 0.5 > y > -1.5: goalOK = False
    if x > 3.5 or x < -3.5 or y > 3.5 or y < -3.5: goalOK = False
    return goalOK

# 🌟 修复点 1: 稳定的 numpy 降采样
def binning(data, quantity=20):
    data = np.array(data)
    data[np.isinf(data)] = 10.0 # 太远测不到，设为10米安全距离
    data[np.isnan(data)] = 0.0  # 👈 修改这里！太近测不到(报错)，直接判定为0米(撞车)！
    
    split_data = np.array_split(data, quantity)
    bins = [np.min(chunk) for chunk in split_data]
    return np.array(bins, dtype=np.float32)
    
    # 使用 split 或者简单的 reshaping 来求每个区块的最小距离
    # 为了避免无法整除，使用 array_split
    split_data = np.array_split(data, quantity)
    bins = [np.min(chunk) for chunk in split_data]
    return np.array(bins, dtype=np.float32) # 返回一维 (20,) 数组

def launchRVIZ(launchfile):
    port = '11311'
    subprocess.Popen(["roscore", "-p", port])
    print("Roscore launched!")
    rospy.init_node('gym', anonymous=True)
    if launchfile.startswith("/"): fullpath = launchfile
    else: fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
    if not path.exists(fullpath): raise IOError("File " + fullpath + " does not exist")
    subprocess.Popen(["roslaunch", "-p", port, fullpath])
    print("Gazebo launched!")

class GazeboEnv:
    def __init__(self, launchfile):
        self.odomX = 0 
        self.odomY = 0
        self.goalX = 1 
        self.goalY = 0.0
        self.upper = 5.0
        self.lower = -5.0
        self.velodyne_data = np.ones(20) * 10 
        self.set_self_state = ModelState() 
        self.set_self_state.model_name = 'r1' 
        self.set_self_state.pose.position.x = 0. 
        self.set_self_state.pose.position.y = 0.
        self.set_self_state.pose.position.z = 0.
        self.set_self_state.pose.orientation.x = 0.0 
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        self.last_laser = None 
        self.last_odom = None 
        
        self.distOld = math.sqrt((self.odomX - self.goalX)**2 + (self.odomY - self.goalY)**2)
        
        # Velodyne角度分割 (保留你原来的逻辑)
        self.gaps = [[-1.6, -1.57 + 3.14 / 20]]
        for m in range(19): self.gaps.append([self.gaps[m][1], self.gaps[m][1] + 3.14 / 20])
        self.gaps[-1][-1] += 0.03 

        launchRVIZ(launchfile)
        
        # 移除了容易卡死的 pause/unpause physics 服务调用
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    
        self.publisher = rospy.Publisher('vis_mark_array', MarkerArray, queue_size=10)
        self.vel_pub = rospy.Publisher('/r1/cmd_vel', Twist, queue_size=10)
        self.set_state = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=10)

        self.velodyne = rospy.Subscriber('/velodyne_points', PointCloud2, self.velodyne_callback, queue_size=10)
        self.laser = rospy.Subscriber('/r1/front_laser/scan', LaserScan, self.laser_callback, queue_size=10)
        self.odom = rospy.Subscriber('/r1/odom', Odometry, self.odom_callback, queue_size=10)

    def laser_callback(self, scan):
        self.last_laser = scan

    def odom_callback(self, od_data):
        self.last_odom = od_data

    def velodyne_callback(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(20) * 10 
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 
                mag1 = math.sqrt(data[i][0]**2 + data[i][1]**2) 
                if mag1 == 0: continue
                beta = math.acos(dot / mag1) * np.sign(data[i][1]) 
                dist = math.sqrt(data[i][0]**2 + data[i][1]**2 + data[i][2]**2) 

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]: 
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist) 
                        break

    def calculate_observation(self, laser_ranges):
        min_range = 0.22 # 稍微加大一点碰撞体积，防止擦边没判定
        done = False
        col = False
        min_laser = np.min(laser_ranges)

        if min_laser < min_range:
            done = True
            col = True
            
        return done, col, min_laser

    def step(self, act, timestep):
        target = False
        vel_cmd = Twist()
        vel_cmd.linear.x = act[0]
        vel_cmd.angular.z = act[1]
        self.vel_pub.publish(vel_cmd)

        # 🌟 修复点 3: 移除了睡眠和 pause，改为确保获取到新数据
        time.sleep(0.05) # 仅留微小延时让物理引擎走一步

        data = self.last_laser
        dataOdom = self.last_odom
        
        # 处理雷达数据
        if data is not None:
            # 如果你有velodyne，这行被我修改成了更稳定的一维融合
            # 注意：你原来有 velodyne_data，这里优先使用前置激光数据
            laser_state = binning(data.ranges, 20) 
            done, col, min_laser = self.calculate_observation(laser_state)
        else:
            laser_state = np.ones(20) * 10
            done, col, min_laser = False, False, 10

        # 从里程计数据计算机器人朝向
        if dataOdom is not None:
            self.odomX = dataOdom.pose.pose.position.x
            self.odomY = dataOdom.pose.pose.position.y
            quaternion = Quaternion(dataOdom.pose.pose.orientation.w, dataOdom.pose.pose.orientation.x, dataOdom.pose.pose.orientation.y, dataOdom.pose.pose.orientation.z)
            euler = quaternion.to_euler(degrees=False)
            angle = round(euler[2], 4)
        else:
            angle = 0.0

        # 计算距离和角度差
        Dist = math.sqrt((self.odomX - self.goalX)**2 + (self.odomY - self.goalY)**2)
        skewX = self.goalX - self.odomX
        skewY = self.goalY - self.odomY
        mag1 = math.sqrt(skewX**2 + skewY**2)
        
        if mag1 == 0: beta = 0
        else: beta = math.acos(skewX / mag1)
        
        if skewY < 0:
            if skewX < 0: beta = -beta
            else: beta = 0 - beta
            
        beta2 = (beta - angle)
        if beta2 > np.pi: beta2 -= 2 * np.pi
        if beta2 < -np.pi: beta2 += 2 * np.pi

        # 🌟 优化 2: 重新设计的 Reward 函数 (引诱型势能奖励)
        reward = 0
        
        # 1. 距离势能奖励 (靠近给正分，远离给负分，这是小车学会寻路的关键！)
        reward += (self.distOld - Dist) * 100 
        self.distOld = Dist
        
        # 2. 基础动作惩罚 (鼓励走直线，少打方向盘)
        reward -= abs(act[1]) * 0.2
        
        # 3. 避障惩罚 (仅当障碍物很近时才扣分)
        if min_laser < 0.6:
            reward -= (0.6 - min_laser) * 10 # 越近扣分越狠

        if Dist < 0.4:
            target = True
            done = True
            reward += 100 # 到达目标给大奖
            
        if col:
            reward -= 100 # 撞墙重罚
            
        if timestep >= 499:
            reward -= 50  # 超时惩罚

        # 🌟 修复点 2: 严格确保 State 是一个 (24,) 的一维 Numpy 数组
        toGoal = np.array([Dist, beta2, act[0], act[1]], dtype=np.float32)
        state = np.concatenate([laser_state, toGoal])

        return state, reward, done, target

    def reset(self):
        rospy.wait_for_service('/gazebo/reset_world')
        try: self.reset_proxy()
        except rospy.ServiceException as e: print("reset call failed")
        
        # 强制停止小车
        vel_cmd = Twist()
        vel_cmd.linear.x = 0; vel_cmd.angular.z = 0
        self.vel_pub.publish(vel_cmd)

        angle = np.random.uniform(-np.pi, np.pi) 
        quaternion = Quaternion.from_euler(0., 0., angle) 
        
        object_state = self.set_self_state
        chk = False
        while not chk: 
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            chk = check_pos(x, y)

        object_state.pose.position.x = x
        object_state.pose.position.y = y
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state) 

        self.odomX = object_state.pose.position.x 
        self.odomY = object_state.pose.position.y

        self.change_goal() 
        self.random_box() 

        # Rviz 目标点可视化
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom" # 修改为全局坐标系
        marker.id = 0
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.05
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goalX
        marker.pose.position.y = self.goalY
        marker.pose.position.z = 0
        markerArray.markers.append(marker)
        self.publisher.publish(markerArray)

        self.distOld = math.sqrt((self.odomX - self.goalX)**2 + (self.odomY - self.goalY)**2)

        time.sleep(0.1) # 等待环境稳定
        
        data = None
        while data is None:
            try: data = rospy.wait_for_message('/r1/front_laser/scan', LaserScan, timeout=2.0)
            except: pass
            
        laser_state = binning(data.ranges, 20)

        Dist = math.sqrt((self.odomX - self.goalX)**2 + (self.odomY - self.goalY)**2)
        skewX = self.goalX - self.odomX
        skewY = self.goalY - self.odomY
        mag1 = math.sqrt(skewX**2 + skewY**2)
        if mag1 == 0: beta = 0
        else: beta = math.acos(skewX / mag1)

        if skewY < 0:
            if skewX < 0: beta = -beta
            else: beta = 0 - beta
            
        beta2 = (beta - angle)
        if beta2 > np.pi: beta2 -= 2 * np.pi
        if beta2 < -np.pi: beta2 += 2 * np.pi
        
        toGoal = np.array([Dist, beta2, 0.0, 0.0], dtype=np.float32)
        state = np.concatenate([laser_state, toGoal])
        
        return state

    def change_goal(self):
        if self.upper < 10: self.upper += 0.004
        if self.lower > -10: self.lower -= 0.004
        
        gOK = False
        while not gOK:
            self.goalX = self.odomX + np.random.uniform(self.upper, self.lower)
            self.goalY = self.odomY + np.random.uniform(self.upper, self.lower)
            gOK = check_pos(self.goalX, self.goalY)

    def random_box(self):
        for i in range(4):
            name = 'cardboard_box_' + str(i)
            x, y, chk = 0, 0, False
            while not chk:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                chk = check_pos(x, y)
                d1 = math.sqrt((x - self.odomX)**2 + (y - self.odomY)**2)
                d2 = math.sqrt((x - self.goalX)**2 + (y - self.goalY)**2)
                if d1 < 1.5 or d2 < 1.5: chk = False
                    
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)