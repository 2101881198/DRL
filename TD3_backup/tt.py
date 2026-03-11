graph TD
    %% 定义样式
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef process fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef attention fill:#ffe0b2,stroke:#f57c00,stroke-width:2px;
    classDef fusion fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef output fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    %% 输入部分
    subgraph Inputs [输入层]
        HS[("历史状态序列<br/>(Batch, 10, 24)")]:::input
        CS[("当前状态<br/>(Batch, 24)")]:::input
    end

    %% 分支1：历史特征处理
    subgraph HistoryBranch [历史特征提取分支]
        HS --> L1["全连接层 (Linear)<br/>24 → 256<br/>ReLU"]:::process
        L1 --> GRU["GRU 层<br/>256 → 256"]:::process
        
        %% Attention 模块
        subgraph AttentionModule [Attention 注意力机制]
            GRU --> AttCalc1["计算得分 u = tanh(x·W)"]:::attention
            AttCalc1 --> AttCalc2["计算权重 att = u·U"]:::attention
            AttCalc2 --> Softmax["Softmax 归一化"]:::attention
            Softmax --> WeightedSum["加权求和 (Sum)"]:::attention
            GRU --> WeightedSum
        end
        
        WeightedSum --"(Batch, 256)"--> L2["全连接层<br/>256 → 128<br/>ReLU + Dropout"]:::process
        L2 --"历史特征 (128维)"--> HistFeat[Feature H]
    end

    %% 分支2：当前特征处理
    subgraph CurrentBranch [当前状态提取分支]
        CS --> L3["全连接层<br/>24 → 384<br/>ReLU + Dropout"]:::process
        L3 --"当前特征 (384维)"--> CurFeat[Feature C]
    end

    %% 融合与输出
    subgraph OutputHead [融合决策层]
        HistFeat --> Concat["拼接 (Concatenate)<br/>128 + 384 = 512"]:::fusion
        CurFeat --> Concat
        Concat --> FinalL["全连接层<br/>512 → 2"]:::output
        FinalL --> Tanh["Tanh 激活函数"]:::output
    end

    Tanh --> Action[("动作输出<br/>[线速度, 角速度]<br/>(Batch, 2)")]:::output
