import numpy as np
import torch.nn as nn
import torch
import pandas as pd
import torch.nn.functional as F



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float32)
torch.cuda.manual_seed_all(0)




# class Tailed_FocalLoss(nn.Module):
#     def __init__(self, threshold, beta=2, gamma=2, reduction:str='mean'):
#         super().__init__()
#         self.beta = beta
#         self.gamma = gamma
#         self.reduction = reduction
#         self.threshold = threshold
#
#
#     def forward(self, probs, labels):
#         probs = F.softmax(probs, dim=1)
#         Pt_index = labels.squeeze()
#         Pt = probs[np.arange(len(Pt_index)), Pt_index]
#         FL_loss = -((1-Pt) ** self.gamma) * torch.log2(Pt)
#         gate = torch.where(Pt_index >= self.threshold, 1, 0).to(DEVICE)
#         Pt_dash = Pt**gate
#         loss = FL_loss + (-torch.log2(Pt_dash**self.beta))
#
#         if self.reduction=='mean':
#             return torch.mean(loss)
#         elif self.reduction == 'sum':
#             return torch.sum(loss)
# class Tailed_FocalLoss(nn.Module):
#     def __init__(self, class_counts, n_classes, gamma=2.0, beta=3.0, ts=0.9, device='cuda'):
#         """
#         Args:
#             class_counts (list or np.array): 训练集中每个类别的样本数量。
#             n_classes (int): 类别总数 (HetDDI 中 DrugBank 为 86)。
#             gamma (float): Focal Loss 聚焦参数，默认 2.0。
#             beta (float): 尾部增强参数，默认 3.0。
#             ts (float): 划分尾部类别的累积频率阈值，默认 0.9。
#             device (str): 设备。
#         """
#         super().__init__()
#         self.gamma = gamma
#         self.beta = beta
#         self.ts = ts
#         self.n_classes = n_classes
#         self.device = device
#
#         # 自动生成尾部掩码 (Tail Mask)
#         # 1 表示该类别是尾部类别，0 表示头部类别
#         self.tail_mask = self._generate_tail_mask(class_counts).to(self.device)
#
#     def _generate_tail_mask(self, class_counts):
#         if isinstance(class_counts, list):
#             class_counts = np.array(class_counts)
#
#         total_samples = sum(class_counts)
#
#         # 1. 获取排序后的索引 (降序: 样本最多的排前面)
#         sorted_indices = np.argsort(class_counts)[::-1]
#         sorted_counts = class_counts[sorted_indices]
#
#         # 2. 计算累积频率
#         cum_counts = np.cumsum(sorted_counts)
#         cum_ratios = cum_counts / total_samples
#
#         # 3. 找到尾部类别（累积频率超过 Ts 的部分被视为尾部）
#         # 这里 logic 是: 如果前 N 个类占了 90% (ts=0.9) 的数据，那么剩下的类就是尾部
#         is_tail_sorted = cum_ratios > self.ts
#
#         # 4. 映射回原始 Label ID
#         mask = torch.zeros(self.n_classes)
#         for i, is_tail in enumerate(is_tail_sorted):
#             if is_tail:
#                 original_class_id = sorted_indices[i]
#                 mask[original_class_id] = 1.0
#
#         return mask
#
#     def forward(self, logits, labels):
#         # logits: (Batch_Size, N_Classes) 未经过 softmax
#         # labels: (Batch_Size)
#
#         probs = F.softmax(logits, dim=1)
#
#         # 获取对应真实标签的概率 Pt
#         Pt = probs.gather(1, labels.view(-1, 1)).view(-1)
#
#         # 防止 log(0)
#         Pt = torch.clamp(Pt, min=1e-8, max=1.0)
#
#         # 标准 Focal Loss 部分
#         # 使用 torch.log (ln) 或 torch.log2 均可，这里保持你代码风格使用 log2，但标准实现多用 ln
#         log_pt = torch.log(Pt)
#         focal_term = -((1 - Pt) ** self.gamma) * log_pt
#
#         # 尾部增强部分
#         # 获取当前 batch 每个样本对应的 mask (1=Tail, 0=Head)
#         current_tail_mask = self.tail_mask[labels]
#
#         # 论文公式: TFL = FL + (- log(Pt ^ (beta * mask)))
#         # - log(Pt ^ (beta * mask)) = - (beta * mask) * log(Pt)
#         tail_term = - self.beta * current_tail_mask * log_pt
#
#         loss = focal_term + tail_term
#
#         return torch.mean(loss)
class Tailed_FocalLoss(nn.Module):
    def __init__(self, class_counts, n_classes, gamma=2.0, beta=3.0, ts=0.9, device='cuda'):
        """
        Tailed Focal Loss [CIKE 2024]

        Args:
            class_counts (list or np.array): 训练集中每个类别的样本数量 (1D array, index=class_id, value=count)。
            n_classes (int): 类别总数 (例如 HetDDI 中 DrugBank 为 86)。
            gamma (float): Focal Loss 聚焦参数，默认 2.0。
            beta (float): 尾部增强参数，默认 3.0。
            ts (float): 划分尾部类别的累积频率阈值，默认 0.9。
            device (str): 计算设备 ('cpu' 或 'cuda:0' 等)。
        """
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.ts = ts
        self.n_classes = n_classes
        self.device = device

        # 自动生成尾部掩码 (Tail Mask)
        # 1 表示该类别是尾部类别，0 表示头部类别
        # 生成后直接移动到指定设备
        mask = self._generate_tail_mask(class_counts)
        self.tail_mask = mask.to(self.device)

    def _generate_tail_mask(self, class_counts):
        """
        根据样本数量分布生成尾部掩码
        """
        # 确保输入是 numpy array
        if isinstance(class_counts, list):
            class_counts = np.array(class_counts)
        elif isinstance(class_counts, torch.Tensor):
            class_counts = class_counts.cpu().numpy()

        total_samples = np.sum(class_counts)

        # 1. 获取排序后的索引 (降序: 样本最多的排前面)
        # argsort 返回的是索引，即 Label ID
        sorted_indices = np.argsort(class_counts)[::-1]
        sorted_counts = class_counts[sorted_indices]

        # 2. 计算累积频率 (Cumulative Frequency)
        cum_counts = np.cumsum(sorted_counts)
        cum_ratios = cum_counts / total_samples

        # 3. 找到尾部类别
        # 逻辑: 累积频率超过 Ts (例如 0.9) 之后的所有类别被视为尾部
        is_tail_sorted = cum_ratios > self.ts

        # 4. 映射回原始 Label ID
        # 初始化全 0 (默认为 Head)
        mask = torch.zeros(self.n_classes)

        for i, is_tail in enumerate(is_tail_sorted):
            if is_tail:
                # 找到排序后第 i 个位置对应的原始类别 ID
                original_class_id = sorted_indices[i]
                mask[original_class_id] = 1.0

        return mask

    def forward(self, logits, labels):
        """
        Args:
            logits: 模型输出 (Batch_Size, N_Classes), 未经过 Softmax
            labels: 真实标签 (Batch_Size)
        """
        # 计算 Softmax 概率
        probs = F.softmax(logits, dim=1)

        # 获取对应真实标签的概率 Pt
        # gather: 选取 target 对应的概率值
        Pt = probs.gather(1, labels.view(-1, 1)).view(-1)

        # 防止 log(0) 导致 NaN
        Pt = torch.clamp(Pt, min=1e-8, max=1.0)

        # === Term 1: Standard Focal Loss ===
        # FL = - (1-Pt)^gamma * log(Pt)
        log_pt = torch.log(Pt)
        focal_term = -((1 - Pt) ** self.gamma) * log_pt

        # === Term 2: Tail Correction ===
        # 获取当前 batch 中每个样本对应的 tail_mask (1=Tail, 0=Head)
        current_tail_mask = self.tail_mask[labels]

        # 论文公式 TFL = FL - log(Pt ^ (beta * mask))
        # - log(x^n) = - n * log(x)
        # 所以 Tail Term = - beta * mask * log(Pt)
        tail_term = - self.beta * current_tail_mask * log_pt

        # 总 Loss
        loss = focal_term + tail_term

        return torch.mean(loss)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction:str='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
    
    
    def forward(self, probs, labels):
        '''
        formula for focal loss: average(-alpha*(1-Pt)log(Pt))
        my probs = [[],[],[]]
        tagert = [[31],[11],[3]]
        '''
        probs = F.softmax(probs, dim=1)
        Pt_index = labels.squeeze()
        Pt = probs[np.arange(len(Pt_index)), Pt_index]
        loss = -((1-Pt) ** self.gamma) * torch.log2(Pt)
        
        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


class WCELoss(nn.Module):
    def __init__(self, train_data_frq:pd.Series):
        super().__init__()  
        self.weights = torch.tensor(np.sum(train_data_frq.values) / train_data_frq.values).to(DEVICE).to(torch.float32)
        # self.wceloss = nn.CrossEntropyLoss(weight=weight, reduction='mean')
        
    def forward(self, probs, labels):
        weight = self.weights[labels]
        probs = F.softmax(probs, dim=1)
        Pt_index = labels.squeeze()
        Pt = probs[np.arange(len(Pt_index)), Pt_index]
        loss = - weight * torch.log2(Pt)
        return torch.mean(loss)


class ClassBalancedLoss(nn.Module):
    def __init__(self, train_data_frq:pd.Series, loss_type="softmax", beta=0.999, gamma=1):
        super().__init__()  
        self.ny = train_data_frq
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma

    def forward(self, probs, labels): 
        power_term = self.ny.loc[labels.cpu().detach().numpy()]
        denominator = 1.0 - np.power(self.beta, power_term)
        weights = torch.tensor((1.0 - self.beta) / np.array(denominator)).to(DEVICE)
        probs = F.softmax(probs, dim=1)
        Pt_index = labels.squeeze()
        Pt = probs[np.arange(len(Pt_index)), Pt_index]
        
        if self.loss_type == "focal":
            cb_loss = - weights * ((1-Pt) ** self.gamma) * torch.log2(Pt)
        
        elif self.loss_type == "softmax":
            cb_loss = - weights * torch.log2(Pt)
            
        return torch.mean(cb_loss)



class BalancedSoftmax(nn.Module):
    """
    Balanced Softmax Loss
    """
    def __init__(self, train_data_frq):
        super().__init__()
        self.sample_per_class = torch.tensor(train_data_frq)

    def forward(self, logits, label, reduction='mean'):
        spc = self.sample_per_class.type_as(logits)
        spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
        logits = logits + spc.log()
        loss = F.cross_entropy(input=logits, target=label, reduction=reduction)
        return loss




class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super( ).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, probs, labels):
        index = torch.zeros_like(probs, dtype=torch.uint8)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = probs - batch_m
        output = torch.where(index, x_m, probs)
        return F.cross_entropy(self.s*output, labels, weight=self.weight)

def get_class_counts(samples, num_classes):
    arr = np.asarray(samples)
    counts = np.zeros(num_classes, dtype=np.int64)
    if arr.size == 0:
        return counts

    labels = np.asarray(arr[:, 2], dtype=np.int64)
    valid_mask = (labels >= 0) & (labels < num_classes)
    labels = labels[valid_mask]
    if labels.size:
        bincount = np.bincount(labels, minlength=num_classes)
        counts[: len(bincount)] = bincount[:num_classes]
    return counts

class CE_TailedFocalLoss(nn.Module):
    def __init__(
        self,
        class_counts,
        n_classes,
        gamma=2.0,
        beta=1.5,
        ts=0.85,
        ce_weight=0.7,
        tail_weight=0.3,
        device="cuda",
    ):
        super(CE_TailedFocalLoss, self).__init__()

        self.ce_loss = nn.CrossEntropyLoss()

        self.tail_loss = Tailed_FocalLoss(
            class_counts=class_counts,
            n_classes=n_classes,
            gamma=gamma,
            beta=beta,
            ts=ts,
            device=device,
        )

        self.ce_weight = ce_weight
        self.tail_weight = tail_weight

    def forward(self, pred, target):
        loss_ce = self.ce_loss(pred, target)
        loss_tail = self.tail_loss(pred, target)

        loss = self.ce_weight * loss_ce + self.tail_weight * loss_tail

        return loss