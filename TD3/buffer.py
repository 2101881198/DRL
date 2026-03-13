import random
import numpy as np
import torch

class ReplayBuffer(object):

    def __init__(self, random_seed=123):

        self.max_size = 1000000
        self.count = 0      # 记录当前已存数据总量
        self.ptr = 0        # 🌟 新增：环形写入指针 (Write Pointer)
        
        self.state_dim = 24
        self.action_dim = 2
        
        self.S_BUF = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.NS_BUF = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.A_BUF = np.zeros((self.max_size, self.action_dim), dtype=np.float32)
        self.R_BUF = np.zeros(self.max_size, dtype=np.float32)
        self.DONE_BUF = np.zeros(self.max_size, dtype=np.float32)
        
        np.random.seed(random_seed) # 统一使用 numpy 随机种子

    def add(self, state, action, reward, done, next_state):
        # 🌟 O(1) 复杂度的环形插入，拒绝内存大搬运！
        self.S_BUF[self.ptr] = state
        self.A_BUF[self.ptr] = action
        self.R_BUF[self.ptr] = reward
        self.NS_BUF[self.ptr] = next_state
        self.DONE_BUF[self.ptr] = done

        # 指针向前走，到头了就绕回 0
        self.ptr = (self.ptr + 1) % self.max_size
        self.count = min(self.count + 1, self.max_size)

    def sample_batch(self, batch_size = 16, max_hisLen = 10):
        
        idxs = []
        # 🌟 1. 安全采样逻辑：确保采样的历史序列不会跨越当前的写入指针（避免新老数据混合）
        while len(idxs) < batch_size:
            idx = np.random.randint(0, self.count)
            if self.count == self.max_size:
                # 如果池子满了，判断 idx 往回倒推 max_hisLen 步是否会撞上写入指针
                dist_to_ptr = (idx - self.ptr) % self.max_size
                if dist_to_ptr < max_hisLen:
                    continue # 序列不纯洁，丢弃重抽
            else:
                # 还没满时，idx 必须大于历史长度
                if idx < max_hisLen:
                    continue
            idxs.append(idx)
            
        idxs = np.array(idxs)
        
        # 初始化 Batch
        HS_BATCH = np.zeros([batch_size, max_hisLen, self.state_dim], dtype=np.float32)
        HNS_BATCH = np.zeros([batch_size, max_hisLen, self.state_dim], dtype=np.float32)
        HA_BATCH = np.zeros([batch_size, max_hisLen, self.action_dim], dtype=np.float32)
        HNA_BATCH = np.zeros([batch_size, max_hisLen, self.action_dim], dtype=np.float32)
        HSL_BATCH = max_hisLen * np.ones(batch_size, dtype=np.float32)
        HNSL_BATCH = max_hisLen * np.ones(batch_size, dtype=np.float32)

        for i, id in enumerate(idxs):
            # 🌟 2. 利用取模运算轻松获取环形数组的历史索引
            seq_idxs = np.arange(id - max_hisLen, id) % self.max_size
            
            # 寻找序列中是否包含 done 标志
            dones = self.DONE_BUF[seq_idxs]
            done_locs = np.where(dones == 1)[0]
            
            # 如果中间有回合结束的标志，我们要进行截断和边缘填充 (Edge Padding)
            if len(done_locs) > 0:
                # 真正的起点在最后一个 done 的下一个位置
                cut_point = done_locs[-1] + 1
                
                # 修改有效长度
                his_seg_len = max_hisLen - cut_point
                HSL_BATCH[i] = his_seg_len
                HNSL_BATCH[i] = his_seg_len
                
                # 边缘填充：把前面的无效位置，全部用 cut_point 的第一个有效状态填充（相当于原地等待）
                seq_idxs[:cut_point] = seq_idxs[cut_point]

            # 🌟 3. 并行切片，抛弃 for 循环和 if/else 填充，速度极致提升！
            HS_BATCH[i] = self.S_BUF[seq_idxs]
            HA_BATCH[i] = self.A_BUF[seq_idxs]
            
            # 由于 NS_BUF 严格平齐 S_BUF，直接用同一组索引即可获取完全对应的 Next_State！
            HNS_BATCH[i] = self.NS_BUF[seq_idxs]
            
            # HNA_BATCH (Next Action) 同样平移即可。用取模避免越界。
            next_action_idxs = (seq_idxs + 1) % self.max_size
            HNA_BATCH[i] = self.A_BUF[next_action_idxs]

        batch = {
            'state': self.S_BUF[idxs],
            'next_state': self.NS_BUF[idxs],
            'action': self.A_BUF[idxs],
            'reward': self.R_BUF[idxs],
            'done': self.DONE_BUF[idxs],
            'h_state': HS_BATCH,
            'h_action': HA_BATCH,
            'h_next_state': HNS_BATCH,
            'h_next_action': HNA_BATCH,
            'h_state_length': HSL_BATCH,
            'h_next_state_length': HNSL_BATCH
        }

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}