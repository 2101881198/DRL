import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    
    def __init__(self, state_dim = 24):
        super(Actor, self).__init__()
        self.cat_len = 128
        
        # 优化点: 最后一层去掉 ReLU，保留连续特征空间
        self.lidar_dim = state_dim - 4 if state_dim > 4 else state_dim
        self.lidar_encoder = nn.Sequential(
            nn.Linear(self.lidar_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.lidar_dim) 
        ).to(device)

        # 历史特征处理
        self.HFC1 = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU())
        self.LSTM = nn.GRU(256, 256, batch_first=True)

        # 优化点: 改进的 Attention 参数 (Query, Key, Value)
        # 用当前特征(Query)去查历史特征(Key)
        self.W_q = nn.Linear(3 * self.cat_len, 256) # 当前特征维度 -> 256
        self.W_k = nn.Linear(256, 256)              # 历史特征维度 -> 256
        self.W_v = nn.Linear(256, 256)              # 历史特征维度 -> 256
        self.scale = torch.sqrt(torch.FloatTensor([256])).to(device)

        self.HFC2 = nn.Sequential(nn.Linear(256, self.cat_len), nn.ReLU(), nn.Dropout())

        # 当前特征处理
        self.CFC = nn.Sequential(nn.Linear(state_dim, 3*self.cat_len), nn.ReLU(), nn.Dropout())
        
        # 最终输出层
        self.FinalFC = nn.Sequential(nn.Linear(4*self.cat_len, 2), nn.Tanh())

    def process_state(self, state):
        is_seq = state.dim() == 3
        if is_seq:
            B, L, D = state.size()
            state = state.view(B * L, D)
        
        if state.size(1) > 4:
            lidar = state[:, :-4]              
            robot = state[:, -4:]              
            lidar_feat = self.lidar_encoder(lidar) 
            full_feat = torch.cat([lidar_feat, robot], dim=1) 
        else:
            full_feat = state

        if is_seq:
            full_feat = full_feat.view(B, L, -1)
            
        return full_feat

    def forward(self, state, his_state, his_len):
        
        state = self.process_state(state)
        his_state = self.process_state(his_state)

        # 1. 处理当前数据 (先提出来，作为 Attention 的 Query)
        cur_output = self.CFC(state) # [B, 384]

        # 2. 处理历史数据
        his_input = self.HFC1(his_state)                        
        his_out_lstm, _ = self.LSTM(his_input) # [B, Seq, 256]
        
        # 3. 改进的 Attention (Scaled Dot-Product)
        # Query: 当前状态特征; Key/Value: LSTM输出的历史特征
        Q = self.W_q(cur_output).unsqueeze(1)  # [B, 1, 256]
        K = self.W_k(his_out_lstm)             # [B, Seq, 256]
        V = self.W_v(his_out_lstm)             # [B, Seq, 256]

        # 计算权重: (Q * K^T) / sqrt(d_k)
        att_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale # [B, 1, Seq]
        att_weights = F.softmax(att_scores, dim=-1)               # [B, 1, Seq]
        
        # 融合特征: Weights * Value
        att_out = torch.bmm(att_weights, V).squeeze(1)            # [B, 256]
        
        his_output = self.HFC2(att_out)                           # [B, 128]

        # 4. 合并历史和当前数据
        final_input = torch.cat([his_output, cur_output], dim = -1)  # [B, 512]
        action = self.FinalFC(final_input)                           # [B, 2]   
        
        return action

class Critic(nn.Module):

    def __init__(self, state_dim = 24, action_dim = 2):
        super(Critic, self).__init__()
        self.cat_len = 128
        
        self.lidar_dim = state_dim - 4 if state_dim > 4 else state_dim
        self.lidar_encoder = nn.Sequential(
            nn.Linear(self.lidar_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.lidar_dim)
        ).to(device)

        # Q1 网络结构 ====================================================
        self.HS1 = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU())
        self.HS_LSTM = nn.GRU(256, 256, batch_first=True)
        
        self.W_q1 = nn.Linear(3 * self.cat_len, 256)
        self.W_k1 = nn.Linear(256, 256)
        self.W_v1 = nn.Linear(256, 256)
        self.scale = torch.sqrt(torch.FloatTensor([256])).to(device)

        self.HS2 = nn.Sequential(nn.Linear(256, self.cat_len), nn.ReLU())
        self.SA1 = nn.Sequential(nn.Linear(state_dim + action_dim, 3*self.cat_len), nn.ReLU())
        self.Final1 = nn.Sequential(nn.Linear(4*self.cat_len, 64), nn.ReLU(), nn.Linear(64 , 1))

        # Q2 网络结构 ====================================================
        self._HS1 = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU())
        self._HS_LSTM = nn.GRU(256, 256, batch_first=True)
        
        self.W_q2 = nn.Linear(3 * self.cat_len, 256)
        self.W_k2 = nn.Linear(256, 256)
        self.W_v2 = nn.Linear(256, 256)

        self._HS2 = nn.Sequential(nn.Linear(256, self.cat_len), nn.ReLU())
        self._SA2 = nn.Sequential(nn.Linear(state_dim + action_dim, 3*self.cat_len), nn.ReLU())
        self._Final2 = nn.Sequential(nn.Linear(4*self.cat_len, 64), nn.ReLU(), nn.Linear(64, 1))

    def process_state(self, state):
        is_seq = state.dim() == 3
        if is_seq:
            B, L, D = state.size()
            state = state.view(B * L, D)
        
        if state.size(1) > 4:
            lidar = state[:, :-4]              
            robot = state[:, -4:]              
            lidar_feat = self.lidar_encoder(lidar) 
            full_feat = torch.cat([lidar_feat, robot], dim=1) 
        else:
            full_feat = state

        if is_seq:
            full_feat = full_feat.view(B, L, -1)
            
        return full_feat

    def forward(self, state, action, his_state, his_action, his_len):
        
        state = self.process_state(state)
        his_state = self.process_state(his_state)

        # 提取当前的 state-action 联合特征 (用作 Query)
        sa = torch.cat([state, action], dim = -1)
        
        cur_out1 = self.SA1(sa)
        cur_out2 = self._SA2(sa)

        # =========== 计算 Q1 的历史特征 ===========
        hs1 = self.HS1(his_state)
        hs_lstm1, _ = self.HS_LSTM(hs1)

        Q1 = self.W_q1(cur_out1).unsqueeze(1)
        K1 = self.W_k1(hs_lstm1)
        V1 = self.W_v1(hs_lstm1)
        att_scores1 = torch.bmm(Q1, K1.transpose(1, 2)) / self.scale
        att_out1 = torch.bmm(F.softmax(att_scores1, dim=-1), V1).squeeze(1)
        hist_out1 = self.HS2(att_out1)

        # =========== 计算 Q2 的历史特征 ===========
        hs2 = self._HS1(his_state)
        hs_lstm2, _ = self._HS_LSTM(hs2)

        Q2 = self.W_q2(cur_out2).unsqueeze(1)
        K2 = self.W_k2(hs_lstm2)
        V2 = self.W_v2(hs_lstm2)
        att_scores2 = torch.bmm(Q2, K2.transpose(1, 2)) / self.scale
        att_out2 = torch.bmm(F.softmax(att_scores2, dim=-1), V2).squeeze(1)
        hist_out2 = self._HS2(att_out2)
        
        # =========== 融合并输出 Q1, Q2 ===========
        final1 = torch.cat([hist_out1, cur_out1], dim = -1)
        final2 = torch.cat([hist_out2, cur_out2], dim = -1)

        q1 = self.Final1(final1).squeeze(-1)
        q2 = self._Final2(final2).squeeze(-1)

        return q1, q2


class TD3(object):
    
    def __init__(self, state_dim=24, action_dim=2):
        
        self.actor = Actor(state_dim).to(device)
        self.actor_target = Actor(state_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4) # 建议显式设定 LR

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4) # Critic 的 LR 通常稍大

        self.max_action = 1

    def get_action(self, state, his_state, his_len):
        self.actor.eval() 
        
        his_state = torch.Tensor(his_state).view(1, his_state.shape[0], his_state.shape[1]).float().to(device)
        his_len = torch.Tensor([his_len]).float().to(device)
        
        with torch.no_grad():
            episode_per_step_action = self.actor(
                torch.as_tensor(state, dtype=torch.float32).view(1, -1).to(device), 
                his_state, 
                his_len
            ).cpu().data.numpy().flatten()
            
        self.actor.train()
        
        return episode_per_step_action                                                                   

    def train(self, replay_buffer, iterations, discount = 0.99, 
                tau = 0.005, policy_noise = 0.2, noise_clip = 0.5, policy_freq = 2):
        
        for it in range(iterations):
            
            batch = replay_buffer.sample_batch()
            state = batch['state'].to(device)
            next_state = batch['next_state'].to(device)
            action = batch['action'].to(device)
            reward = batch['reward'].to(device)
            done = batch['done'].to(device)
            h_state = batch['h_state'].to(device)
            h_next_state = batch['h_next_state'].to(device)
            h_action = batch['h_action'].to(device)
            h_next_action = batch['h_next_action'].to(device)
            h_state_len = batch['h_state_length'].to(device)
            h_next_state_len = batch['h_next_state_length'].to(device)            

            # 更新 Critic -------------------------------------------------------------
            with torch.no_grad():
                next_action = self.actor_target(next_state, h_next_state, h_next_state_len)
                noise = action.data.normal_(0, policy_noise)
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

                target_Q1, target_Q2 = self.critic_target(next_state, next_action, h_next_state, 
                                                            h_next_action, h_next_state_len)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (1 - done) * discount * target_Q

            current_Q1, current_Q2 = self.critic(state, action, h_state, h_action, h_state_len)

            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # 延迟更新 Actor ----------------------------------------------------------
            if it % policy_freq == 0:

                # 注意：计算 Actor Loss 时，只用 Q1 即可，并且冻结 Critic 参数以加速
                for params in self.critic.parameters():
                    params.requires_grad = False

                actor_action = self.actor(state, h_state, h_state_len)
                actor_loss, _ = self.critic(state, actor_action, h_state, h_action, h_state_len)
                actor_loss = -actor_loss.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # 恢复 Critic 参数的梯度
                for params in self.critic.parameters():
                    params.requires_grad = True

                # 软更新目标网络
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))