import random
import numpy as np
import torch

class ReplayBuffer(object):

    def __init__(self, random_seed=123):

        self.max_size = 1000000
        self.count = 0      # 记录当前已存数据总量
        self.ptr = 0        # 环形写入指针
        
        self.state_dim = 24
        self.action_dim = 2
        
        self.S_BUF = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.NS_BUF = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.A_BUF = np.zeros((self.max_size, self.action_dim), dtype=np.float32)
        self.R_BUF = np.zeros(self.max_size, dtype=np.float32)
        self.DONE_BUF = np.zeros(self.max_size, dtype=np.float32)
        
        np.random.seed(random_seed) 

    def add(self, state, action, reward, done, next_state):
        # O(1) 复杂度的环形插入，拒绝内存大搬运，极速写入！
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
        # 安全采样逻辑：确保序列绝对纯洁，不跨越任何断层
        while len(idxs) < batch_size:
            idx = np.random.randint(0, self.count)
            
            # 1. 如果池子没满，避开最开头的那几步（防止往回倒推取到末尾的 0）
            if self.count < self.max_size and idx < max_hisLen - 1:
                continue
                
            # 2. 如果池子满了，避开写入指针所在的新旧数据断层
            if self.count == self.max_size:
                dist_to_ptr = (idx - self.ptr) % self.max_size
                if dist_to_ptr < max_hisLen - 1:
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
            
            # 👇 修复核心：序列必须包含 id 本身！长度刚好是 max_hisLen
            seq_idxs = np.arange(id - max_hisLen + 1, id + 1) % self.max_size
            
            # 我们只检查 'id' 之前的步骤是否发生了死亡/截断，不检查 'id' 本身
            dones = self.DONE_BUF[seq_idxs[:-1]]
            done_locs = np.where(dones == 1)[0]
            
            # 如果前面有死亡记录，说明这段历史包含了上一个回合的废数据，需要截断
            if len(done_locs) > 0:
                # 真正的有效历史起点，在最后一次死亡的下一帧
                cut_point = done_locs[-1] + 1
                
                # 记录有效长度
                his_seg_len = max_hisLen - cut_point
                HSL_BATCH[i] = his_seg_len
                HNSL_BATCH[i] = his_seg_len
                
                # 边缘填充 (Edge Padding)：把前面无效的废数据，全用第一帧有效数据覆盖
                seq_idxs[:cut_point] = seq_idxs[cut_point]
            else:
                HSL_BATCH[i] = max_hisLen
                HNSL_BATCH[i] = max_hisLen

            # 并行提取矩阵数据
            HS_BATCH[i] = self.S_BUF[seq_idxs]
            HA_BATCH[i] = self.A_BUF[seq_idxs]
            HNS_BATCH[i] = self.NS_BUF[seq_idxs]
            
            # Next action 平移取模即可
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