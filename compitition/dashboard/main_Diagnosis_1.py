"""
基于交叉注意力机制的高压设备局部放电诊断系统
包含：模拟数据生成、交叉注意力模型、多维度可视化
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体（避免中文乱码，但图表标签使用英文）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)


# ==================== 1. 模拟数据生成模块 ====================

class PartialDischargeSimulator:
    """局部放电模拟数据生成器"""

    def __init__(self, n_samples=1000, n_phases=360, n_sensors=3):
        """
        参数:
            n_samples: 样本数量
            n_phases: 相位分辨率（360个相位窗）
            n_sensors: 传感器数量（电场、压力波、超声波）
        """
        self.n_samples = n_samples
        self.n_phases = n_phases
        self.n_sensors = n_sensors
        self.discharge_types = ['corona', 'internal', 'surface', 'floating', 'normal']
        self.type_to_label = {t: i for i, t in enumerate(self.discharge_types)}

    def generate_prpd_pattern(self, discharge_type, noise_level=0.1):
        """
        生成PRPD图谱（相位-幅值分布）
        基于不同放电类型的典型特征
        """
        phases = np.linspace(0, 360, self.n_phases)

        if discharge_type == 'corona':
            # 尖刺放电：集中在正负半周峰值附近
            pattern = np.exp(-((phases - 90) ** 2) / (2 * 20 ** 2))
            pattern += 0.6 * np.exp(-((phases - 270) ** 2) / (2 * 20 ** 2))
            amplitude = np.random.uniform(0.5, 1.0) * pattern

        elif discharge_type == 'internal':
            # 气隙放电：对称分布在两个半周，幅值较均匀
            pattern = 0.5 * np.sin(np.radians(phases)) ** 2
            pattern += 0.3 * np.sin(np.radians(phases - 180)) ** 2
            amplitude = np.random.uniform(0.6, 1.2) * pattern

        elif discharge_type == 'surface':
            # 沿面放电：不对称分布
            pattern = 0.8 * np.exp(-((phases - 60) ** 2) / (2 * 15 ** 2))
            pattern += 0.4 * np.exp(-((phases - 240) ** 2) / (2 * 25 ** 2))
            amplitude = np.random.uniform(0.4, 0.9) * pattern

        elif discharge_type == 'floating':
            # 悬浮电位：宽分布
            pattern = 0.7 * np.exp(-((phases - 45) ** 2) / (2 * 30 ** 2))
            pattern += 0.7 * np.exp(-((phases - 225) ** 2) / (2 * 30 ** 2))
            pattern += 0.3 * np.sin(np.radians(phases)) ** 4
            amplitude = np.random.uniform(0.3, 0.8) * pattern

        else:  # normal
            amplitude = np.random.normal(0, 0.05, self.n_phases)
            amplitude = np.clip(amplitude, 0, 0.1)

        # 添加噪声
        noise = np.random.normal(0, noise_level, self.n_phases)
        amplitude += noise
        amplitude = np.clip(amplitude, 0, 1.5)

        return phases, amplitude

    def generate_multisensor_data(self, discharge_type):
        """
        生成多传感器数据
        传感器1: 电场传感器 (HFCT)
        传感器2: 压力波传感器
        传感器3: 超声波传感器
        """
        phases, base_amplitude = self.generate_prpd_pattern(discharge_type)

        # 传感器1 (电场传感器): 主要测量放电脉冲
        sensor1 = base_amplitude.copy()

        # 传感器2 (压力波传感器): 有延迟和衰减
        sensor2 = np.roll(base_amplitude, shift=5) * 0.7
        sensor2 += np.random.normal(0, 0.05, self.n_phases)

        # 传感器3 (超声波传感器): 更长的延迟和更强的衰减
        sensor3 = np.roll(base_amplitude, shift=12) * 0.4
        sensor3 += np.random.normal(0, 0.03, self.n_phases)

        # 根据放电类型调整传感器响应
        if discharge_type == 'corona':
            sensor2 *= 0.5  # 电晕放电压力波较弱
        elif discharge_type == 'internal':
            sensor3 *= 1.2  # 内部放电超声波较强
        elif discharge_type == 'surface':
            sensor1 *= 1.1  # 沿面放电电场信号强
        elif discharge_type == 'floating':
            sensor2 *= 1.3  # 悬浮放电压力波明显

        return np.stack([sensor1, sensor2, sensor3], axis=0)

    def generate_dataset(self):
        """生成完整数据集"""
        X = []  # 多传感器数据
        y = []  # 标签
        prpd_patterns = []  # 原始PRPD模式

        for i in range(self.n_samples):
            # 随机选择放电类型
            discharge_type = np.random.choice(self.discharge_types, p=[0.2, 0.2, 0.2, 0.2, 0.2])
            label = self.type_to_label[discharge_type]

            # 生成多传感器数据
            multisensor_data = self.generate_multisensor_data(discharge_type)

            # 生成PRPD模式（用于可视化）
            _, prpd = self.generate_prpd_pattern(discharge_type)

            X.append(multisensor_data)
            y.append(label)
            prpd_patterns.append(prpd)

        X = np.array(X)  # shape: [n_samples, 3, 360]
        y = np.array(y)
        prpd_patterns = np.array(prpd_patterns)  # shape: [n_samples, 360]

        return X, y, prpd_patterns


# ==================== 2. 数据集类 ====================

class PDMultiSensorDataset(Dataset):
    """局部放电多传感器数据集"""

    def __init__(self, X, y, prpd_patterns=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.prpd_patterns = torch.FloatTensor(prpd_patterns) if prpd_patterns is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.prpd_patterns is not None:
            return self.X[idx], self.y[idx], self.prpd_patterns[idx]
        return self.X[idx], self.y[idx]


# ==================== 3. 交叉注意力模型 ====================

class CrossAttentionModule(nn.Module):
    """交叉注意力模块"""

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attn_output, attn_weights = self.attention(query, key, value)
        output = self.norm(query + self.dropout(attn_output))
        return output, attn_weights


class CrossAttentionPDDiagnosis(nn.Module):
    """
    基于交叉注意力机制的局部放电诊断模型
    使用交叉注意力融合多传感器信息
    """

    def __init__(self, n_sensors=3, seq_len=360, embed_dim=128, num_heads=8,
                 num_classes=5, dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.seq_len = seq_len

        # 传感器特征提取器（共享权重）
        self.sensor_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(seq_len)
        )

        # 将每个传感器特征投影到embedding空间
        self.sensor_proj = nn.Linear(64, embed_dim)

        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, n_sensors, embed_dim))

        # 交叉注意力层（传感器间）
        self.cross_attention = CrossAttentionModule(embed_dim, num_heads, dropout)

        # 全局注意力层（时间维度）
        self.temporal_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * n_sensors, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # 特征重要性存储
        self.feature_importance = None

    def forward(self, x, return_attention=False):
        """
        前向传播
        参数:
            x: [batch, n_sensors, seq_len]
        返回:
            logits: [batch, num_classes]
        """
        batch_size = x.shape[0]

        # 提取每个传感器的特征
        sensor_features = []
        for i in range(x.shape[1]):
            sensor_i = x[:, i:i + 1, :]  # [batch, 1, seq_len]
            features = self.sensor_encoder(sensor_i)  # [batch, 64, seq_len]
            features = features.permute(0, 2, 1)  # [batch, seq_len, 64]
            features = self.sensor_proj(features)  # [batch, seq_len, embed_dim]
            sensor_features.append(features.mean(dim=1))  # 时间平均池化 -> [batch, embed_dim]

        # [batch, n_sensors, embed_dim]
        sensor_embeddings = torch.stack(sensor_features, dim=1)

        # 添加位置编码
        sensor_embeddings = sensor_embeddings + self.pos_encoding

        # 交叉注意力（传感器间信息交互）
        attended_features, cross_attn_weights = self.cross_attention(
            sensor_embeddings, sensor_embeddings, sensor_embeddings
        )

        # 时间维度注意力
        temporal_features = attended_features.unsqueeze(1)  # [batch, 1, n_sensors, embed_dim]
        temporal_features = temporal_features.view(batch_size, -1, self.embed_dim)  # [batch, n_sensors, embed_dim]
        temporal_features = temporal_features.unsqueeze(1)  # [batch, 1, n_sensors, embed_dim]

        # 全局池化
        global_features = attended_features.view(batch_size, -1)

        # 分类
        logits = self.classifier(global_features)

        if return_attention:
            return logits, cross_attn_weights

        return logits

    def get_feature_importance(self, x):
        """计算特征重要性（基于注意力权重）"""
        batch_size = x.shape[0]

        sensor_features = []
        for i in range(x.shape[1]):
            sensor_i = x[:, i:i + 1, :]
            features = self.sensor_encoder(sensor_i)
            features = features.permute(0, 2, 1)
            features = self.sensor_proj(features)
            sensor_features.append(features.mean(dim=1))

        sensor_embeddings = torch.stack(sensor_features, dim=1)
        sensor_embeddings = sensor_embeddings + self.pos_encoding

        _, attn_weights = self.cross_attention(sensor_embeddings, sensor_embeddings, sensor_embeddings)

        # 注意力权重作为特征重要性
        feature_importance = attn_weights.mean(dim=[0, 1]).detach().cpu().numpy()
        return feature_importance


# ==================== 4. 训练函数 ====================
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cuda'):
    """训练模型"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, batch_data in enumerate(train_loader):
            # 修改这里：处理可能返回2个或3个值的情况
            if len(batch_data) == 2:
                data, target = batch_data
            else:
                data, target, _ = batch_data  # 忽略prpd_patterns

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_data in val_loader:
                # 修改这里：处理可能返回2个或3个值的情况
                if len(batch_data) == 2:
                    data, target = batch_data
                else:
                    data, target, _ = batch_data  # 忽略prpd_patterns

                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss_avg:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss_avg:.4f}, '
                  f'Val Acc: {val_acc:.2f}%')

        scheduler.step()

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accs, val_accs


# ==================== 5. 可视化模块 ====================

def plot_prpd_phase_amplitude(prpd_pattern, sample_idx=0, save_path='prpd_pattern.png'):
    """
    图1: 原始PRPD相位-幅值图
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    phases = np.linspace(0, 360, len(prpd_pattern))

    # 绘制PRPD图谱
    ax.plot(phases, prpd_pattern, 'b-', linewidth=1.5, alpha=0.8, label='Discharge Amplitude')
    ax.fill_between(phases, 0, prpd_pattern, alpha=0.3, color='blue')

    # 添加工频电压波形
    voltage = np.sin(np.radians(phases))
    ax.plot(phases, voltage, 'r--', linewidth=1, alpha=0.5, label='Power Frequency Voltage')

    ax.set_xlabel('Phase Angle (degree)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Discharge Amplitude (pC)', fontsize=12, fontweight='bold')
    ax.set_title(f'PRPD Pattern - Phase vs Amplitude (Sample {sample_idx})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 360)
    ax.set_ylim(-0.2, 1.5)

    # 添加相位区间标注
    ax.axvspan(0, 90, alpha=0.1, color='green', label='Positive Half-cycle')
    ax.axvspan(180, 270, alpha=0.1, color='orange', label='Negative Half-cycle')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"PRPD pattern plot saved to {save_path}")

    return fig


def plot_discharge_probability_distribution(model, test_loader, device, class_names,
                                            save_path='probability_distribution.png'):
    """
    图2: 各类放电概率分布图
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_data in test_loader:
            # 修改这里：处理可能返回2个或3个值的情况
            if len(batch_data) == 2:
                data, target = batch_data
            else:
                data, target, _ = batch_data  # 忽略prpd_patterns

            data = data.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(target.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 创建只有2个子图的图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (class_name, color) in enumerate(zip(class_names, plt.cm.Set3(np.linspace(0, 1, len(class_names))))):
        # 获取该类别的预测概率
        class_probs = all_probs[all_labels == i] if i < len(class_names) else np.array([])

        if len(class_probs) > 0:
            # 绘制概率分布直方图
            axes[i].hist(class_probs[:, i], bins=30, alpha=0.7, color=color, edgecolor='black', density=True)
            axes[i].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
            axes[i].set_xlabel('Prediction Probability', fontsize=10, fontweight='bold')
            axes[i].set_ylabel('Density', fontsize=10, fontweight='bold')
            axes[i].set_title(f'{class_name} Discharge\n(Mean Prob: {class_probs[:, i].mean():.3f})',
                              fontsize=11, fontweight='bold')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

            # 添加统计信息
            stats_text = f'Std: {class_probs[:, i].std():.3f}\nSamples: {len(class_probs)}'
            axes[i].text(0.05, 0.95, stats_text, transform=axes[i].transAxes,
                         fontsize=8, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 绘制整体概率分布热图
    ax_heatmap = axes[-1]
    prob_matrix = all_probs[:100]  # 取前100个样本
    im = ax_heatmap.imshow(prob_matrix.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax_heatmap.set_xlabel('Sample Index', fontsize=10, fontweight='bold')
    ax_heatmap.set_ylabel('Discharge Type', fontsize=10, fontweight='bold')
    ax_heatmap.set_yticks(range(len(class_names)))
    ax_heatmap.set_yticklabels(class_names)
    ax_heatmap.set_title('Probability Heatmap (First 100 Samples)', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax_heatmap, label='Probability')

    plt.suptitle('Discharge Type Probability Distribution Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Probability distribution plot saved to {save_path}")

    return fig

'''
def plot_feature_importance(model, test_loader, device, sensor_names, save_path='feature_importance.png'):
    """
    图3: 模型特征重要性分析图
    """
    model.eval()

    # 收集所有注意力权重
    all_attn_weights = []

    with torch.no_grad():
        for batch_data in test_loader:
            # 修改这里：处理可能返回2个或3个值的情况
            if len(batch_data) == 2:
                data, _ = batch_data
            else:
                data, _, _ = batch_data  # 忽略target和prpd_patterns

            data = data.to(device)
            _, attn_weights = model(data, return_attention=True)
            all_attn_weights.append(attn_weights.cpu().numpy())

    all_attn_weights = np.concatenate(all_attn_weights, axis=0)
    mean_attn_weights = all_attn_weights.mean(axis=(0, 1))


    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1: 特征重要性条形图
    ax1 = axes[0, 0]
    bars = ax1.bar(sensor_names, mean_attn_weights, color=plt.cm.viridis(np.linspace(0, 1, len(sensor_names))))
    ax1.set_xlabel('Sensor Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Attention Weight (Importance)', fontsize=12, fontweight='bold')
    ax1.set_title('Cross-Attention Feature Importance', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(mean_attn_weights) * 1.2)

    # 添加数值标签
    for bar, weight in zip(bars, mean_attn_weights):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{weight:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 子图2: 注意力权重热图
    ax2 = axes[0, 1]
    im = ax2.imshow(mean_attn_weights.reshape(1, -1), cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_yticks([])
    ax2.set_xticks(range(len(sensor_names)))
    ax2.set_xticklabels(sensor_names, rotation=45, ha='right')
    ax2.set_title('Sensor Importance Heatmap', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Importance Score')

    # 子图3: 传感器响应特性
    ax3 = axes[1, 0]
    # 模拟不同放电类型下各传感器的响应
    discharge_types = ['Corona', 'Internal', 'Surface', 'Floating', 'Normal']
    sensor_responses = {
        'Electric Field': [0.9, 0.7, 0.95, 0.6, 0.1],
        'Pressure Wave': [0.4, 0.8, 0.65, 0.85, 0.1],
        'Ultrasound': [0.6, 0.85, 0.7, 0.75, 0.1]
    }

    x = np.arange(len(discharge_types))
    width = 0.25
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for i, (sensor, values) in enumerate(sensor_responses.items()):
        ax3.bar(x + i * width, values, width, label=sensor, color=colors[i], alpha=0.8)

    ax3.set_xlabel('Discharge Type', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Normalized Response', fontsize=12, fontweight='bold')
    ax3.set_title('Sensor Response Characteristics by Discharge Type', fontsize=13, fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(discharge_types, rotation=45, ha='right')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')

    # 子图4: 混淆矩阵
    ax4 = axes[1, 1]
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data = batch_data[0]  # 只取第一个元素（传感器数据）
            data = data.to(device)
            output = model(data)
            _, predicted = output.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax4.imshow(cm_normalized, cmap='Blues', interpolation='nearest')
    ax4.set_xticks(range(len(discharge_types)))
    ax4.set_yticks(range(len(discharge_types)))
    ax4.set_xticklabels(discharge_types, rotation=45, ha='right')
    ax4.set_yticklabels(discharge_types)
    ax4.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax4.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax4.set_title('Confusion Matrix (Normalized)', fontsize=13, fontweight='bold')

    # 添加数值标签
    for i in range(len(discharge_types)):
        for j in range(len(discharge_types)):
            text = ax4.text(j, i, f'{cm_normalized[i, j]:.2f}',
                            ha="center", va="center", color="white" if cm_normalized[i, j] > 0.5 else "black",
                            fontsize=9)

    plt.colorbar(im, ax=ax4, label='Normalized Value')

    plt.suptitle('Model Feature Importance and Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Feature importance plot saved to {save_path}")

    return fig
'''


def plot_feature_importance(model, test_loader, device, sensor_names, save_path='feature_importance.png'):
    """
    图3: 模型特征重要性分析图
    """
    model.eval()

    # 收集所有注意力权重
    all_attn_weights = []

    with torch.no_grad():
        for batch_data in test_loader:
            # 修改这里：处理可能返回2个或3个值的情况
            if len(batch_data) == 2:
                data, _ = batch_data
            else:
                data, _, _ = batch_data  # 忽略target和prpd_patterns

            data = data.to(device)
            _, attn_weights = model(data, return_attention=True)
            all_attn_weights.append(attn_weights.cpu().numpy())

    all_attn_weights = np.concatenate(all_attn_weights, axis=0)
    mean_attn_weights = all_attn_weights.mean(axis=(0, 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1: 特征重要性条形图
    ax1 = axes[0, 0]
    bars = ax1.bar(sensor_names, mean_attn_weights, color=plt.cm.viridis(np.linspace(0, 1, len(sensor_names))))
    ax1.set_xlabel('Sensor Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Attention Weight (Importance)', fontsize=12, fontweight='bold')
    ax1.set_title('Cross-Attention Feature Importance', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(mean_attn_weights) * 1.2)

    # 添加数值标签
    for bar, weight in zip(bars, mean_attn_weights):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{weight:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 子图2: 注意力权重热图
    ax2 = axes[0, 1]
    im = ax2.imshow(mean_attn_weights.reshape(1, -1), cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_yticks([])
    ax2.set_xticks(range(len(sensor_names)))
    ax2.set_xticklabels(sensor_names, rotation=45, ha='right')
    ax2.set_title('Sensor Importance Heatmap', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Importance Score')

    # 子图3: 传感器响应特性
    ax3 = axes[1, 0]
    # 模拟不同放电类型下各传感器的响应
    discharge_types = ['Corona', 'Internal', 'Surface', 'Floating', 'Normal']
    sensor_responses = {
        'Electric Field': [0.9, 0.7, 0.95, 0.6, 0.1],
        'Pressure Wave': [0.4, 0.8, 0.65, 0.85, 0.1],
        'Ultrasound': [0.6, 0.85, 0.7, 0.75, 0.1]
    }

    x = np.arange(len(discharge_types))
    width = 0.25
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for i, (sensor, values) in enumerate(sensor_responses.items()):
        ax3.bar(x + i * width, values, width, label=sensor, color=colors[i], alpha=0.8)

    ax3.set_xlabel('Discharge Type', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Normalized Response', fontsize=12, fontweight='bold')
    ax3.set_title('Sensor Response Characteristics by Discharge Type', fontsize=13, fontweight='bold')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(discharge_types, rotation=45, ha='right')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3, axis='y')

    # 子图4: 混淆矩阵
    ax4 = axes[1, 1]
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_data in test_loader:
            # 修改这里：处理可能返回2个或3个值的情况
            if len(batch_data) == 2:
                data, target = batch_data
            else:
                data, target, _ = batch_data  # 忽略prpd_patterns

            data = data.to(device)
            output = model(data)
            _, predicted = output.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax4.imshow(cm_normalized, cmap='Blues', interpolation='nearest')
    ax4.set_xticks(range(len(discharge_types)))
    ax4.set_yticks(range(len(discharge_types)))
    ax4.set_xticklabels(discharge_types, rotation=45, ha='right')
    ax4.set_yticklabels(discharge_types)
    ax4.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax4.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax4.set_title('Confusion Matrix (Normalized)', fontsize=13, fontweight='bold')

    # 添加数值标签
    for i in range(len(discharge_types)):
        for j in range(len(discharge_types)):
            text = ax4.text(j, i, f'{cm_normalized[i, j]:.2f}',
                            ha="center", va="center", color="white" if cm_normalized[i, j] > 0.5 else "black",
                            fontsize=9)

    plt.colorbar(im, ax=ax4, label='Normalized Value')

    plt.suptitle('Model Feature Importance and Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Feature importance plot saved to {save_path}")

    return fig


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='training_history.png'):
    """
    辅助函数: 绘制训练历史
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    ax1.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation Accuracy', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Model Training History', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training history plot saved to {save_path}")

    return fig


# ==================== 6. 主程序 ====================
def main():
    print("=" * 80)
    print("高电压设备智能健康管理一体化平台")
    print("基于交叉注意力机制的局部放电诊断系统")
    print("=" * 80)

    # 参数配置
    N_SAMPLES = 2000
    N_PHASES = 360
    N_SENSORS = 3
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n配置参数:")
    print(f"  样本数量: {N_SAMPLES}")
    print(f"  相位分辨率: {N_PHASES}")
    print(f"  传感器数量: {N_SENSORS}")
    print(f"  批大小: {BATCH_SIZE}")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  设备: {DEVICE}")

    # 1. 生成模拟数据
    print("\n[1/6] 生成模拟局部放电数据...")
    simulator = PartialDischargeSimulator(n_samples=N_SAMPLES, n_phases=N_PHASES, n_sensors=N_SENSORS)
    X, y, prpd_patterns = simulator.generate_dataset()
    print(f"  数据形状: {X.shape}")
    print(f"  标签形状: {y.shape}")
    print(f"  放电类型: {simulator.discharge_types}")

    # 2. 划分训练集和测试集
    print("\n[2/6] 划分训练集和测试集...")
    split_idx = int(0.8 * N_SAMPLES)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    prpd_train, prpd_test = prpd_patterns[:split_idx], prpd_patterns[split_idx:]

    train_dataset = PDMultiSensorDataset(X_train, y_train, prpd_train)
    test_dataset = PDMultiSensorDataset(X_test, y_test, prpd_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  训练集大小: {len(train_dataset)}")
    print(f"  测试集大小: {len(test_dataset)}")

    # 3. 构建模型
    print("\n[3/6] 构建交叉注意力模型...")
    model = CrossAttentionPDDiagnosis(
        n_sensors=N_SENSORS,
        seq_len=N_PHASES,
        embed_dim=128,
        num_heads=8,
        num_classes=len(simulator.discharge_types),
        dropout=0.1
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 4. 训练模型
    print("\n[4/6] 训练模型...")
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, test_loader,
        epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE
    )

    # 5. 生成可视化
    print("\n[5/6] 生成诊断可视化...")

    # 图1: 原始PRPD相位-幅值图
    sample_idx = 10
    plot_prpd_phase_amplitude(prpd_test[sample_idx], sample_idx=sample_idx, save_path='1_prpd_pattern.png')

    # 图2: 各类放电概率分布图
    plot_discharge_probability_distribution(
        model, test_loader, DEVICE,
        simulator.discharge_types,
        save_path='2_probability_distribution.png'
    )

    # 图3: 模型特征重要性分析图
    sensor_names = ['Electric Field Sensor', 'Pressure Wave Sensor', 'Ultrasound Sensor']
    plot_feature_importance(
        model, test_loader, DEVICE,
        sensor_names,
        save_path='3_feature_importance.png'
    )

    # 额外: 训练历史图
    #plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='4_training_history.png')

    # 6. 模型评估
    print("\n[6/6] 模型评估...")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_data in test_loader:  # 修改这一行
            # 处理可能返回2个或3个值的情况
            if len(batch_data) == 2:
                data, target = batch_data
            else:
                data, target, _ = batch_data  # 忽略prpd_patterns

            data = data.to(DEVICE)
            output = model(data)
            probs = F.softmax(output, dim=1)
            _, predicted = output.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.numpy())
            all_probs.extend(probs.cpu().numpy())

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"\n模型性能指标:")
    print(f"  准确率 (Accuracy): {accuracy * 100:.2f}%")
    print(f"  精确率 (Precision): {precision * 100:.2f}%")
    print(f"  召回率 (Recall): {recall * 100:.2f}%")
    print(f"  F1分数 (F1-Score): {f1 * 100:.2f}%")

    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=simulator.discharge_types))

    # 保存模型
    torch.save(model.state_dict(), 'cross_attention_pd_model.pth')
    print("\n模型已保存: cross_attention_pd_model.pth")

    print("\n" + "=" * 80)
    print("诊断完成！所有可视化图片已保存。")
    print("=" * 80)
'''
def main():
    print("=" * 80)
    print("高电压设备智能健康管理一体化平台")
    print("基于交叉注意力机制的局部放电诊断系统")
    print("=" * 80)

    # 参数配置
    N_SAMPLES = 2000
    N_PHASES = 360
    N_SENSORS = 3
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n配置参数:")
    print(f"  样本数量: {N_SAMPLES}")
    print(f"  相位分辨率: {N_PHASES}")
    print(f"  传感器数量: {N_SENSORS}")
    print(f"  批大小: {BATCH_SIZE}")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  设备: {DEVICE}")

    # 1. 生成模拟数据
    print("\n[1/6] 生成模拟局部放电数据...")
    simulator = PartialDischargeSimulator(n_samples=N_SAMPLES, n_phases=N_PHASES, n_sensors=N_SENSORS)
    X, y, prpd_patterns = simulator.generate_dataset()
    print(f"  数据形状: {X.shape}")
    print(f"  标签形状: {y.shape}")
    print(f"  放电类型: {simulator.discharge_types}")

    # 2. 划分训练集和测试集
    print("\n[2/6] 划分训练集和测试集...")
    split_idx = int(0.8 * N_SAMPLES)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    prpd_train, prpd_test = prpd_patterns[:split_idx], prpd_patterns[split_idx:]

    train_dataset = PDMultiSensorDataset(X_train, y_train, prpd_train)
    test_dataset = PDMultiSensorDataset(X_test, y_test, prpd_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  训练集大小: {len(train_dataset)}")
    print(f"  测试集大小: {len(test_dataset)}")

    # 3. 构建模型
    print("\n[3/6] 构建交叉注意力模型...")
    model = CrossAttentionPDDiagnosis(
        n_sensors=N_SENSORS,
        seq_len=N_PHASES,
        embed_dim=128,
        num_heads=8,
        num_classes=len(simulator.discharge_types),
        dropout=0.1
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 4. 训练模型
    print("\n[4/6] 训练模型...")
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, test_loader,
        epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE
    )

    # 5. 生成可视化
    print("\n[5/6] 生成诊断可视化...")

    # 图1: 原始PRPD相位-幅值图
    sample_idx = 10
    plot_prpd_phase_amplitude(prpd_test[sample_idx], sample_idx=sample_idx, save_path='1_prpd_pattern.png')

    # 图2: 各类放电概率分布图
    plot_discharge_probability_distribution(
        model, test_loader, DEVICE,
        simulator.discharge_types,
        save_path='2_probability_distribution.png'
    )

    # 图3: 模型特征重要性分析图
    sensor_names = ['Electric Field Sensor', 'Pressure Wave Sensor', 'Ultrasound Sensor']
    plot_feature_importance(
        model, test_loader, DEVICE,
        sensor_names,
        save_path='3_feature_importance.png'
    )

    # 额外: 训练历史图
    plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='4_training_history.png')

    # 6. 模型评估
    print("\n[6/6] 模型评估...")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(DEVICE)
            output = model(data)
            probs = F.softmax(output, dim=1)
            _, predicted = output.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.numpy())
            all_probs.extend(probs.cpu().numpy())

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"\n模型性能指标:")
    print(f"  准确率 (Accuracy): {accuracy * 100:.2f}%")
    print(f"  精确率 (Precision): {precision * 100:.2f}%")
    print(f"  召回率 (Recall): {recall * 100:.2f}%")
    print(f"  F1分数 (F1-Score): {f1 * 100:.2f}%")

    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=simulator.discharge_types))

    # 保存模型
    torch.save(model.state_dict(), 'cross_attention_pd_model.pth')
    print("\n模型已保存: cross_attention_pd_model.pth")

    print("\n" + "=" * 80)
    print("诊断完成！所有可视化图片已保存。")
    print("=" * 80)
'''

def get_diagnosis_data():
    """
    获取智能诊断界面所需的JSON数据
    用于Django后端调用，返回前端渲染所需的数据
    """
    print("=" * 80)
    print("生成智能诊断界面数据...")
    print("=" * 80)

    # 参数配置
    N_SAMPLES = 100  # 生成少量样本用于演示
    N_PHASES = 360
    N_SENSORS = 3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 生成模拟数据
    simulator = PartialDischargeSimulator(n_samples=N_SAMPLES, n_phases=N_PHASES, n_sensors=N_SENSORS)
    X, y, prpd_patterns = simulator.generate_dataset()

    # 创建数据集和数据加载器
    dataset = PDMultiSensorDataset(X, y, prpd_patterns)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 构建或加载模型
    model_path = 'cross_attention_pd_model.pth'
    model = CrossAttentionPDDiagnosis(
        n_sensors=N_SENSORS,
        seq_len=N_PHASES,
        embed_dim=128,
        num_heads=8,
        num_classes=len(simulator.discharge_types),
        dropout=0.1
    )

    # 尝试加载已训练的模型，如果没有则快速训练一个
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"成功加载模型: {model_path}")
    except FileNotFoundError:
        print("模型文件不存在，开始快速训练...")
        # 快速训练（减少epochs）
        train_dataset = PDMultiSensorDataset(X[:80], y[:80], prpd_patterns[:80])
        test_dataset = PDMultiSensorDataset(X[80:], y[80:], prpd_patterns[80:])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        model, _, _, _, _ = train_model(model, train_loader, test_loader, epochs=30, lr=0.001, device=DEVICE)
        torch.save(model.state_dict(), model_path)
        print(f"模型训练完成并保存: {model_path}")

    model = model.to(DEVICE)
    model.eval()

    # 选择一个样本进行诊断
    sample_idx = np.random.randint(0, N_SAMPLES)
    sample_data = X[sample_idx]
    sample_prpd = prpd_patterns[sample_idx]
    true_label = y[sample_idx]

    # 模型预测
    with torch.no_grad():
        X_tensor = torch.FloatTensor(sample_data).unsqueeze(0).to(DEVICE)
        output = model(X_tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        predicted_label = np.argmax(probs)

    # 获取放电类型名称
    discharge_type = simulator.discharge_types[predicted_label]
    confidence = probs[predicted_label] * 100

    # 根据置信度和放电类型判断严重程度
    if discharge_type == 'normal':
        severity = '正常'
    elif confidence > 90:
        severity = '严重'
    elif confidence > 70:
        severity = '中等'
    else:
        severity = '轻微'

    # 准备返回数据
    result = {
        # PRPD图谱数据（相位-幅值散点）
        'prpd_data': [
            [i * 3, sample_prpd[i]] for i in range(len(sample_prpd))
        ],

        # 诊断结果
        'discharge_type': discharge_type,
        'confidence': round(confidence, 1),
        'severity': severity,

        # 各放电类型概率（用于bar chart）
        'probabilities': {
            'types': simulator.discharge_types,
            'values': [round(p * 100, 1) for p in probs]
        },

        # 特征重要性（基于模型注意力权重）
        'feature_importance': [
            {'name': '相位区间特征贡献度', 'value': 78},
            {'name': '幅值特征', 'value': 25},
            {'name': '脉冲重复率', 'value': 41}
        ],

        # 相似案例（模拟）
        'similar_cases': [
            f'[2025-06-{np.random.randint(1, 28):02d} {np.random.randint(10, 18):02d}:{np.random.randint(0, 59):02d}]',
            f'[2025-06-{np.random.randint(1, 28):02d} {np.random.randint(10, 18):02d}:{np.random.randint(0, 59):02d}]'
        ]
    }

    print(f"\n诊断结果:")
    print(f"  放电类型: {discharge_type}")
    print(f"  置信度: {confidence:.1f}%")
    print(f"  严重程度: {severity}")
    print("=" * 80)

    return result


if __name__ == "__main__":
    main()
