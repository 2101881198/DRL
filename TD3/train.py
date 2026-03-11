import time                                                                     # 导入时间模块，用于控制等待和计时
import os                                                                       # 导入操作系统接口模块，用于创建文件夹等操作
import numpy as np                                                              # 导入科学计算库numpy
import torch                                                                    # 导入深度学习框架PyTorch
from env import GazeboEnv                                                       # 从自定义的env模块导入Gazebo环境类
from buffer import ReplayBuffer                                                 # 从自定义的buffer模块导入经验回放池类
# from gru_net import TD3                                                       # （已注释）导入基于GRU的TD3网络
#from td3_net import TD3                                                        # （已注释）导入标准TD3网络
from attention_net import TD3                                                   # 导入基于注意力机制的TD3网络
import rospy                                                                    # 导入ROS Python接口模块
from sensor_msgs.msg import LaserScan                                           # 从ROS中导入激光雷达消息类型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           # 判断是否有GPU可用，如果有则使用CUDA，否则使用CPU
seed = 0                                                                        # 设置全局随机种子
eval_freq = 5e3                                                                 # 每隔多少个step（步数）做一次策略评估
eval_ep = 10                                                                    # 每次策略评估运行多少个回合（episode）
max_ep_step = 500                                                               # 单个回合（episode）的最大步数限制
max_timesteps = 5e6                                                             # 整个训练过程的最大总步数
save_models = True                                                              # 标志位：是否在训练过程中保存模型
max_his_len = 10                                                                # 历史观测序列的最大长度（针对时序网络或注意力机制）
expl_noise = 1.0                                                                # 初始探索噪音的方差/大小
expl_decay_steps = 500000                                                       # 探索噪音衰减完毕所需经历的步数
expl_min = 0.1                                                                  # 探索噪音衰减的最小值
save_reward = -9999                                                             # 初始的最高奖励记录（用于保存最优模型）
discount = 0.99999                                                              # 贝尔曼方程中的折扣因子（Gamma）
tau = 0.005                                                                     # 目标网络软更新的平滑系数
policy_noise = 0.2                                                              # 目标策略平滑正则化中加入的噪音大小
noise_clip = 0.5                                                                # 目标策略噪音的裁剪阈值
policy_freq = 2                                                                 # Actor网络（及其目标网络）的延迟更新频率（每更新几次Critic再更新一次Actor）
file_name = "TD3_velodyne"                                                      # 模型保存的文件名前缀
random_near_obstacle = False                                                    # 标志位：是否在靠近障碍物时触发随机避障动作（通常为后退/转向）

#* 2. 创建存储模型的文件夹
if not os.path.exists("./results"): os.makedirs("./results")                    # 如果不存在结果文件夹，则创建该文件夹（用于保存评估数据图表）
if save_models and not os.path.exists("./pytorch_models"): os.makedirs("./pytorch_models")  # 如果允许保存且不存在pytorch_models文件夹，则创建它
if not os.path.exists("./best_models"): os.makedirs("./best_models")            # 如果不存在保存最佳模型的文件夹，则创建
if not os.path.exists("./final_models"): os.makedirs("./final_models")          # 如果不存在保存最终/最新模型的文件夹，则创建

#* 3. 环境初始化
env = GazeboEnv('multi_robot_scenario.launch')                                  # 实例化Gazebo环境对象，加载对应的ROS launch文件
print("正在等待雷达话题 /r1/front_laser/scan ...")                              # 打印提示信息，等待激光雷达数据接入

try:
    # 设置超时时间为 20 秒，确保雷达有足够时间启动
    rospy.wait_for_message("/r1/front_laser/scan", LaserScan, timeout=20)       # 阻塞等待ROS激光雷达话题的数据，最多等20秒
    print(">>> 成功检测到雷达数据！环境已就绪。")                               # 如果成功接收到消息，打印就绪提示
except rospy.ROSException:
    print(">>> 【严重警告】等待雷达超时！环境可能只能返回 24 维数据。")             # 如果超时未收到数据，捕获异常并打印警告
    print(">>> 请检查 Gazebo 是否崩溃，或话题名是否正确。")                     # 提醒检查仿真环境状态
time.sleep(5)                                                                   # 强制休眠5秒，留出额外的稳定时间
torch.manual_seed(seed)                                                         # 为PyTorch设置随机种子，保障结果可复现
np.random.seed(seed)                                                            # 为numpy设置随机种子，保障结果可复现

#* 6. 初始化训练参数
state_dim = 24                                                                  # 状态维度设为24（通常包含激光雷达降采样数据及目标点相对位置等）
action_dim = 2                                                                  # 动作维度设为2（线速度和角速度）
max_action = 1                                                                  # 网络输出动作的最大绝对值（归一化为1）

network = TD3(state_dim, action_dim)                                            # 初始化TD3算法模型（包含Actor和Critic网络及其优化器）
replay_buffer = ReplayBuffer(seed)                                              # 初始化经验回放池，用于打乱和存储过往转移数据（Transitions）

count_rand_actions = 0                                                          # 记录靠近障碍物时执行强制随机退避动作的剩余步数
random_action = []                                                              # 用于临时存放生成的退避动作的值

evaluations = []                                                                # 列表：记录历次评估的平均奖励
timestep = 0                                                                    # 初始化全局已经执行的训练步数
timesteps_since_eval = 0                                                        # 距离上一次执行验证经历了多少步
episode_num = 0                                                                 # 已开始的回合数
epoch = 1                                                                       # 验证阶段的轮数（记录进行了第几次验证）
done = True                                                                     # 标志位：回合是否结束（初始化为True以便首次进入循环后能马上启动新回合）
total_reward = 0                                                                # 记录累积奖励的临时变量
collide = False                                                                 # 碰撞标志位
start_update_timestep = 100                                                     # 收集到多少步经验后，才开始调用网络更新(train)函数
network_action_timestep = 10000                                                 # 前面多少步完全使用随机动作（纯探索），这里设为1万步后才使用网络输出动作

def evaluate(network, eval_episodes = 10, epoch=0, max_his_len = 10):           # 定义评估函数：在没有探索噪音时测试当前的策略性能
    avg_reward = 0.                                                             # 初始化验证集的所有回合平均奖励累加器
    col = 0                                                                     # 初始化验证回合发生碰撞的次数累加器
    for _ in range(eval_episodes):                                              # 运行指定次数（默认10次）的回合
        count = 0                                                               # 记录当前评估回合走过的步数
        state = env.reset()                                                     # 环境重置，获得初始状态。该状态通常为格式numpy(1, 23/24)
        done = False                                                            # 标志当前验证回合未结束
        EP_HS = np.zeros([max_his_len, 24])                                     # 初始化一个记录历史状态的张量缓冲，以配合时序注意力网络
        EP_HS[::] = list(state)                                                 # 将历史状态全部拉平初始化为初始状态
        EP_HL = 0                                                               # 初始化历史指针（有效历史长度减1）

        while not done and count < 501:                                         # 评估的内部步循环，最多走500步
            action = network.get_action(state, EP_HS, EP_HL)                    # 让TD3模型纯确信地根据当前状态与历史信息推理出连续动作
            a_in = [(action[0] + 1) / 2, action[1]]                             # 将线速度放缩到 [0, 1] 之间（假设网络输出都是[-1,1]），角速度保持[-1, 1]
            
            next_state, reward, done, _ = env.step(a_in, count-1)               # 在仿真环境中执行动作并获取下一状态、单步奖励和回合结束标志
            if EP_HL == max_his_len:                                            # 如果历史状态记录已达到设定的最大长度
                EP_HS[:(max_his_len-1)] = EP_HS[1:]                             # 将后面的历史状态向前移一位
                EP_HS[max_his_len-1] = list(state)                              # 腾出的最后一位存放当刻的状态
            else:                                                               # 如果历史状态未满
                if EP_HL > 1:                                                   # 如果指针大于1（避免越界）
                    EP_HS[-(EP_HL+1)] = list(state)                             # 将当前状态放进历史记录倒数对应的位置
                EP_HL += 1                                                      # 历史计数器递增
            
            state = next_state                                                  # 状态滚动更新为下一步的状态
            avg_reward += reward                                                # 将单步奖励累加到总奖励上
            count += 1                                                          # 步数加1
            if reward < -90:                                                    # 如果奖励小于-90（通常在定义里说明发生了碰撞）
                col += 1                                                        # 碰撞次数加1

    avg_reward /= eval_episodes                                                 # 除以回合总数，得到平均奖励
    avg_col = col/eval_episodes                                                 # 得到碰撞率（平均碰撞次数）
    print("..............................................")                     # 打印分割线，美观控制台输出
    print("Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f" % (eval_episodes, epoch, avg_reward, avg_col)) # 打印该阶段具体的评估效果
    print("..............................................")                     # 打印分割线结尾
    return avg_reward                                                           # 将评估的平均奖励返回，以供决定是否要保存最佳模型

state = env.reset()                                                             # 【主训练开始前】重置环境获取初始的状态

episode_reward = 0                                                              # 初始化当前主回合的累加奖励为0
episode_timesteps = 0                                                           # 初始化当前主回合经历的步数为0
episode_num += 1                                                                # 记录这是第几个回合（初始加到1）
EP_HS = np.zeros([max_his_len, state_dim])                                      # 初始化主训练的历史状态缓冲
EP_HS[::] = list(state)                                                         # 类似地，全部覆盖为初始状态
EP_HL = 0                                                                       # 主训练的历史指针初始化


while timestep < max_timesteps:                                                 # 启动庞大的主训练循环，直到完成规定的总步数
    if expl_noise > expl_min:                                                   # 如果当前的探索噪音大于设定的下限：
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)           # 则进行线性衰减探索噪音
    # according to state choose action 5000                                     # 注释：根据状态选取动作
    if timestep >= network_action_timestep:                                     # 如果经过了“纯随机探索期”（比如前1万步）：
        action = network.get_action(state, EP_HS, EP_HL)                        # 调用神经网络的actor，传入当前状态与历史信息生成基础动作
        action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(-max_action, max_action) # 对策略给出的动作加入高斯噪声并进行截断裁剪以保证不超过边界[-1,1]
    else:                                                                       # 若处于“纯随机探索期”内：
        action = np.random.uniform(-1,1,2)                                      # 完全随机从 [-1, 1] 生成2维动作

    if random_near_obstacle:                                                    # 如果启用了基于规则的靠近障碍物的防卡死机制
        if np.random.uniform(0, 1) > 0.85 and min(state[4:-8]) < 0.6 and count_rand_actions < 1: # 85%概率触发 & 前向激光探测到障碍物很近(<0.6m) & 当前没有正在执行后退动作
            count_rand_actions = np.random.randint(8, 15)                       # 随机选取 8-15 步的后退/转向动作分配给智能体
            random_action = np.random.uniform(-1, 1, 2)                         # 随机生成一个动作组合用于之后的这几步避障动作
        if count_rand_actions > 0:                                              # 如果当前还处于被动避障指令周期中：
            count_rand_actions -= 1                                             # 倒计时步数减1
            action = random_action                                              # 我们强制网络这步走原定生成的随机避障动作
            action[0] = -1                                                      # 并且把线速度设为强制后退(-1相当于0，因为后面会有重映射)

    a_in = [(action[0] + 1) / 2, action[1]]                                     # 进行物理控制前的动作缩放调整，线速度变换为[0, 1]，即只能向前不能向后


    # action into env, get state                                                # 将动作送进真实仿真环境，获取新的状态
    next_state, reward, done, target = env.step(a_in, episode_timesteps)        # 进一步通过Gazebo环境执行动作，并返回最新状态、奖励和结束标志
    episode_reward += reward                                                    # 将这次动作的回报奖励累加入当前轮回总奖励中
    
    episode_timesteps += 1                                                      # 单轮回的计步器自增1
    timestep += 1                                                               # 整个训练的物理总步数自增1
    timesteps_since_eval += 1                                                   # 距离下一次评估考核的步数自增1

    if episode_timesteps == max_ep_step:                                        # 如果达到了单个回合步数上限（比如500步）
        done = False                                                            # 强制将done修改为False来避免吸收状态的影响，因为这并非因为碰撞或到达终点导致的终止

    replay_buffer.add(state, action, reward, done, next_state)                  # 将四元组(状态, 动作, 奖励, 结束旗帜, 下个状态)存进经验池中供以后回放

    if EP_HL == max_his_len:                                                    # 继续维护训练过程中的历史观测缓存序列。如果满载：
        EP_HS[:(max_his_len-1)] = EP_HS[1:]                                     # 弹出最老的数据（索引左移）
        EP_HS[max_his_len-1] = list(state)                                      # 将当前状态插在序列队尾
    else:                                                                       # 序列未满载时：
        if EP_HL > 1:                                                           # 为了避免首次更新覆盖或者越界等奇怪问题
            EP_HS[-(EP_HL+1)] = list(state)                                     # 以负索引方式将当前记录存住
        EP_HL += 1                                                              # 更新指针长度

    state = next_state                                                          # 使用新观测到的状态更新原有状态，进入下一次循环的数据流
    

    if done or episode_timesteps == max_ep_step:                                # 【回合结束逻辑处理】如果发生了碰撞/到达目标（done=True），或者由于步数到了回合强行结束
        
        if timestep >= network_action_timestep:                                 # 输出不同阶段的调试控制台打印。如果已经过了随机摸索期：
            print('\033[1;45m Actor Action Update \033[0m', 'episode_reward:', round(episode_reward, 2), 'evaluation:', timesteps_since_eval) # 紫色背景提示网络介入了控制，并打印本回合的回报
        else:
            if timestep >= start_update_timestep:                               # 如果虽然处于完全随机，但经验池已经足够开始启动神经网络初步更新
                print('\033[1;45m Random Action Update \033[0m', 'evaluation:', timesteps_since_eval)  # 紫色打印提示当前是在更新网络，但采样动作还是随机抽取
            else:
                print('\033[1;46m Data Collection \033[0m')                     # 经验池尚空洞时，打印青色提示当前仅仅是在收集基础数据


    # if timestep >= start_update_timestep and timestep % 50 == 0:              #（此行代码被注释）表示可选择每累积某些步数后才成批训练
        if timestep >= start_update_timestep:                                   # 而目前逻辑：只要步数超过了启动训练的最少经历线，则该回合死亡后立刻启动训练！
            network.train(replay_buffer, 50,                                    # 调用TD3网络内部方法，执行权重更新迭代：采样batch更新（此处的50似乎代表batch或迭代次数）
                        discount, tau, policy_noise, noise_clip, policy_freq)   # 传入所有的TD3特定超参数：贝尔曼折扣、软更新系数、噪音添加限制等
        
        state = env.reset()                                                     # 更新完网络后，把陷入死胡同或者到达重点的环境重置（恢复初始坐标）
        episode_reward = 0                                                      # 并重置该新回合的奖励累积器
        episode_timesteps = 0                                                   # 重置该回合的内部步数
        episode_num += 1                                                        # 总回合数自增计数1
        EP_HS = np.zeros([max_his_len, state_dim])                              # 开个新的空白数组来装新回合的历史观测值
        EP_HS[::] = list(state)                                                 # 把刚才reset得到的全新出场状态填满这些空位
        EP_HL = 0                                                               # 重启历史记录指针

    if timesteps_since_eval >= eval_freq:                                       # 【评估模型分支】如果距上一次评估走过的步数查过了设定频率（例如5千步）
        print("================= Validating =================")                 # 打印开始验证的大横幅
        timesteps_since_eval %= eval_freq                                       # 将验证累加步数余上评估阈值重置
        tmp_reward = evaluate(network, eval_ep, epoch)                          # 调用提前定义好的纯评估函数，对网络进行无探索噪声的测试，得到成绩
        evaluations.append(tmp_reward)                                          # 将得到的评分/平均奖励加入全局评估列表中
        if tmp_reward >= save_reward:                                           # 如果成绩打破了历史最好的记录保存阈值：
            save_reward = tmp_reward                                            # 提高历史最好门槛阈值
            network.save(file_name, directory="./best_models")                  # 调用保存函数，将这套最优网络权重存到最佳模型文件夹（./best_models）下
        network.save(file_name, directory="./final_models")                     # 每隔一段时间，无论这次是否最好，都把当前进度保存到最后模型名下作为一个最新存档
        np.save("./results/%s" % (file_name), evaluations)                      # 利用numpy保存历史以来的测试成绩列表为文件，方便后续画训练曲线图
        epoch += 1                                                              # 将验证阶段的世代计数加上一，为后续日志说明做准备

