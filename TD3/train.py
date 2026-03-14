import time                                                                     
import os                                                                       
import numpy as np                                                              
import torch                                                                    
from env import GazeboEnv                                                       
from buffer import ReplayBuffer                                                 
from attention_net import TD3                                                   
import rospy                                                                    
from sensor_msgs.msg import LaserScan                                           

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           
seed = 0                                                                        
eval_freq = 5e3                                                                 
eval_ep = 10                                                                    
max_ep_step = 500                                                               
max_timesteps = 5e6                                                             
save_models = True                                                              
max_his_len = 10                                                                
expl_noise = 1.0                                                                
expl_decay_steps = 500000                                                       
expl_min = 0.1                                                                  
save_reward = -9999                                                             
discount = 0.99999                                                              
tau = 0.005                                                                     
policy_noise = 0.2                                                              
noise_clip = 0.5                                                                
policy_freq = 2                                                                 
file_name = "TD3_velodyne"                                                      
random_near_obstacle = False                                                    

# 创建存储模型的文件夹
if not os.path.exists("./results"): os.makedirs("./results")                    
if save_models and not os.path.exists("./pytorch_models"): os.makedirs("./pytorch_models")  
if not os.path.exists("./best_models"): os.makedirs("./best_models")            
if not os.path.exists("./final_models"): os.makedirs("./final_models")          

# 环境初始化
env = GazeboEnv('multi_robot_scenario.launch')                                  
print("正在等待雷达话题 /r1/front_laser/scan ...")                              

try:
    rospy.wait_for_message("/r1/front_laser/scan", LaserScan, timeout=20)       
    print(">>> 成功检测到雷达数据！环境已就绪。")                               
except rospy.ROSException:
    print(">>> 【严重警告】等待雷达超时！环境可能只能返回 24 维数据。")             
    print(">>> 请检查 Gazebo 是否崩溃，或话题名是否正确。")                     
time.sleep(5)                                                                   

torch.manual_seed(seed)                                                         
np.random.seed(seed)                                                            

state_dim = 24                                                                  
action_dim = 2                                                                  
max_action = 1                                                                  

network = TD3(state_dim, action_dim)                                            
replay_buffer = ReplayBuffer(seed)                                              

count_rand_actions = 0                                                          
random_action = []                                                              

evaluations = []                                                                
timestep = 0                                                                    
timesteps_since_eval = 0                                                        
episode_num = 0                                                                 
epoch = 1                                                                       
done = True                                                                     
total_reward = 0                                                                
collide = False                                                                 

# 同步收集期和训练开启时间
start_update_timestep = 10000 
network_action_timestep = 10000

def evaluate(network, eval_episodes = 10, epoch=0, max_his_len = 10):           
    avg_reward = 0.                                                             
    col = 0                                                                     
    for _ in range(eval_episodes):                                              
        count = 0                                                               
        state = env.reset()                                                     
        done = False                                                            
        
        # 历史滑动窗口初始化
        EP_HS = np.tile(state, (max_his_len, 1))

        while not done and count < 501:                                         
            action = network.get_action(state, EP_HS, max_his_len)                     
            a_in = [(action[0] + 1) / 2, action[1]]                             
            
            next_state, reward, done, _ = env.step(a_in, count-1)               
            
            # 滑动更新历史
            EP_HS = np.roll(EP_HS, -1, axis=0)
            EP_HS[-1] = next_state
            
            state = next_state                                                  
            avg_reward += reward                                                
            count += 1                                                          
            if reward < -90:                                                    
                col += 1                                                        

    avg_reward /= eval_episodes                                                 
    avg_col = col/eval_episodes                                                 
    print("..............................................")                     
    print("Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f" % (eval_episodes, epoch, avg_reward, avg_col)) 
    print("..............................................")                     
    return avg_reward                                                           

# * ================= 主训练循环开始 ================= *
state = env.reset()                                                             
episode_reward = 0                                                              
episode_timesteps = 0                                                           
episode_num += 1                                                                

EP_HS = np.tile(state, (max_his_len, 1)) 

while timestep < max_timesteps:                                                 
    if expl_noise > expl_min:                                                   
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)           
    
    # 动作选择
    if timestep >= network_action_timestep:                                     
        action = network.get_action(state, EP_HS, max_his_len)                        
        action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(-max_action, max_action) 
    else:                                                                       
        action = np.random.uniform(-1,1,2)                                      

    # 随机避障机制
    if random_near_obstacle:                                                    
        if np.random.uniform(0, 1) > 0.85 and min(state[4:-8]) < 0.6 and count_rand_actions < 1: 
            count_rand_actions = np.random.randint(8, 15)                       
            random_action = np.random.uniform(-1, 1, 2)                         
        if count_rand_actions > 0:                                              
            count_rand_actions -= 1                                             
            action = random_action                                              
            action[0] = -1                                                      

    a_in = [(action[0] + 1) / 2, action[1]]                                     

    # 与环境交互
    next_state, reward, done, target = env.step(a_in, episode_timesteps)        
    episode_reward += reward                                                    
    
    episode_timesteps += 1                                                      
    timestep += 1                                                               
    timesteps_since_eval += 1                                                   

    # 判断是否是因为超时导致的伪死亡
    done_bool = False if episode_timesteps == max_ep_step else done

    # 记录经验 (新版 Buffer 内部会自动回溯历史，传 5 个参数即可)
    replay_buffer.add(state, action, reward, done_bool, next_state) 

    # 更新历史滑动窗口
    EP_HS = np.roll(EP_HS, -1, axis=0)
    EP_HS[-1] = next_state

    state = next_state                                                          

    # 回合结束结算
    if done or episode_timesteps == max_ep_step:                                
        
        if timestep >= network_action_timestep:                                 
            print('\033[1;45m Actor Action Update \033[0m', 'episode_reward:', round(episode_reward, 2), 'evaluation:', timesteps_since_eval) 
        else:
            print('\033[1;46m Data Collection \033[0m')                     

        # 🌟 乾坤大挪移：第一步，立刻重置环境，把小车拉回安全区！
        state = env.reset()                                                     
        EP_HS = np.tile(state, (max_his_len, 1)) # 新回合重新初始化历史

        # 🌟 乾坤大挪移：第二步，让神经网络开始安安静静地更新算力
        if timestep >= start_update_timestep:                                   
            network.train(replay_buffer, episode_timesteps,                     
                        discount, tau, policy_noise, noise_clip, policy_freq)   
        
        # 第三步：重置回合变量
        episode_reward = 0                                                      
        episode_timesteps = 0                                                   
        episode_num += 1                                                        

    # 评估与保存最优模型
    if timesteps_since_eval >= eval_freq:                                       
        print("================= Validating =================")                 
        timesteps_since_eval %= eval_freq                                       
        tmp_reward = evaluate(network, eval_ep, epoch)                          
        evaluations.append(tmp_reward)                                          
        if tmp_reward >= save_reward:                                           
            save_reward = tmp_reward                                            
            network.save(file_name, directory="./best_models")                  
        network.save(file_name, directory="./final_models")                     
        np.save("./results/%s" % (file_name), evaluations)                      
        epoch += 1