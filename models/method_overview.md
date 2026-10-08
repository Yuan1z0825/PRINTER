# Method Overview

本文提出一个整合染色归一化、虚拟染色和多模态融合的计算病理学框架，旨在解决FFPE与新鲜冷冻（FF）样本之间的染色差异问题，并实现鲁棒的组织病理图像分析。

## 1. 整体框架

框架由三个核心模块组成：

1. **染色归一化模块**：基于原型学习的风格迁移网络，将不同染色条件的图像归一化到统一的色彩空间
2. **虚拟染色模块**：配准引导的条件生成网络，实现FFPE到FF风格的转换
3. **多模态融合模块**：染色感知的图神经网络，聚合多尺度、多染色的特征表示

## 2. 染色归一化

### 2.1 方法概述

染色归一化模块采用双分支生成器架构，包含内容编码器、风格编码器和解码器。核心思想是通过风格原型学习实现对目标域染色特性的鲁棒表示。

### 2.2 风格原型学习

维护一组可学习的风格原型 $P = \{p_1, p_2, \ldots, p_K\}$，通过Sinkhorn算法实现风格特征到原型的最优传输分配：

$$\hat{s}_i = \frac{s_i}{\|s_i\|_2}, \quad \hat{p}_j = \frac{p_j}{\|p_j\|_2}$$

$$Q_{ij} = \hat{s}_i^\top \hat{p}_j$$

原型通过动量更新进行优化：

$$p_k^{(t+1)} = \gamma \cdot p_k^{(t)} + (1-\gamma) \cdot \frac{\bar{s}_k}{\|\bar{s}_k\|_2}$$

### 2.3 自适应实例归一化（AdaIN）

风格迁移的核心操作通过AdaIN实现：

$$\mu_c = \frac{1}{HW}\sum_{h,w} f_c(h,w), \quad \sigma_c = \sqrt{\frac{1}{HW}\sum_{h,w}(f_c(h,w) - \mu_c)^2}$$

$$\text{AdaIN}(f_c, \gamma, \beta) = \gamma \odot \frac{f_c - \mu_c}{\sigma_c} + \beta$$

其中 $\gamma$ 和 $\beta$ 由聚合的风格向量通过全连接层生成。

### 2.4 可微分Otsu组织分割

引入可微分的Otsu阈值分割模块，通过可学习的RGB权重和高斯核密度估计实现端到端的组织区域提取：

$$g = \sum_{c} w_c \cdot I_c, \quad T^* = \sum_k \alpha_k \cdot c_k$$

$$M = 1 - \sigma\left(\frac{g - T^*}{\tau}\right)$$

## 3. 虚拟染色与配准

### 3.1 配准引导的生成框架

在染色归一化的基础上，虚拟染色模块引入VoxelMorph变形网络实现空间对齐。变形网络预测稠密位移场 $\phi$：

$$\phi = \mathcal{R}(I_G, I_T), \quad I_{reg} = \mathcal{ST}(I_G, \phi)$$

### 3.2 掩码条件对抗训练

判别器接收图像与组织掩码的拼接作为条件输入：

$$\text{Fake}: [I_{reg}, M_S], \quad \text{Real}: [I_T, M_T]$$

### 3.3 多目标损失函数

总损失函数整合多个监督信号：

$$\mathcal{L}_{total} = \mathcal{L}_{GAN} + \lambda_{NCE}\mathcal{L}_{NCE} + \lambda_{pixel}\mathcal{L}_{pixel} + \lambda_{freq}\mathcal{L}_{freq} + \lambda_{content}\mathcal{L}_{content}$$

其中对比损失 $\mathcal{L}_{NCE}$ 保持空间对应关系，配准损失包含L1重建、互信息正则化和平滑约束：

$$\mathcal{L}_{pixel} = \|I_{reg} - I_T\|_1 - \log(\omega - \text{NMI}(I_{reg}, I_T)) + \mathcal{L}_{smooth}(\phi)$$

## 4. 多模态图神经网络融合

### 4.1 图构建

将WSI构建为图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$，节点表示组织块，边编码空间关系。每个节点关联特征向量、染色属性和随机游走位置编码。

### 4.2 染色感知注意力池化（SAAPooling）

提出染色感知池化机制，在池化过程中保留染色特异性信息。对于每种染色类型 $k$，计算注意力权重并聚合特征：

$$w_k = \frac{\sum_{i: a_i = k} \text{att}_i}{\sum_j \text{att}_j}$$

$$\bar{x}_k = w_k \cdot \text{mean}_{i \in V_k}(h_i), \quad \tilde{x}_k = w_k \cdot \max_{i \in V_k}(h_i)$$

### 4.3 图注意力卷积

采用GATv2卷积层传播信息：

$$h_i^{(l+1)} = \alpha_{ii} W h_i^{(l)} + \sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j^{(l)}$$

$$\alpha_{ij} = \text{softmax}_{j}\left(\text{LeakyReLU}(a^\top [Wh_i \| Wh_j \| W_e e_{ij}])\right)$$

### 4.4 层注意力多尺度融合

多层输出通过自注意力机制自适应融合：

$$Z = [z^{(1)} \| z^{(2)} \| \ldots \| z^{(L)}]$$

$$\text{Attn}(Z) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

### 4.5 下游任务

- **分类任务**：$\hat{y} = \text{softmax}(W_2 \cdot \text{ReLU}(W_1 \cdot z))$
- **生存分析**：$h_t = \sigma(W \cdot z)$，$S(t) = \prod_{k=1}^{t}(1-h_k)$

## 5. 处理流程

```
原始WSI → 染色归一化 → 虚拟染色(FFPE样本) → 特征提取 → 图构建 → 多模态融合 → 临床预测
```

| 模块 | 核心创新 | 技术贡献 |
|------|----------|----------|
| 染色归一化 | 风格原型学习 | Sinkhorn分配+动量更新的可学习原型库 |
| 虚拟染色 | 配准引导生成 | VoxelMorph空间对齐+掩码条件对抗训练 |
| 多模态融合 | 染色感知池化 | 图神经网络+染色加权层级池化+层注意力 |