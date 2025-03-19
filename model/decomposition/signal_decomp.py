import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion of wavelet and HHT decomposition components
    """
    def __init__(self, feature_dim, fusion_type='weighted', window_size=64):
        super(AdaptiveFusion, self).__init__()
        self.feature_dim = feature_dim
        self.fusion_type = fusion_type
        self.window_size = window_size
        
        # 创建统一的特征投影层，确保所有组件映射到相同的隐藏维度
        hidden_dim = 64  # 统一的隐藏维度
        
        # Learnable weights for fusion
        self.wavelet_weight_high = nn.Parameter(torch.ones(1))
        self.hht_weight_high = nn.Parameter(torch.ones(1))
        self.wavelet_weight_low = nn.Parameter(torch.ones(1))
        self.hht_weight_low = nn.Parameter(torch.ones(1))
        
        # Energy consistency regularizer
        self.energy_factor = nn.Parameter(torch.ones(1))
        
        # 统一特征投影，确保维度一致
        self.wavelet_high_proj = nn.Linear(feature_dim, hidden_dim)
        self.hht_high_proj = nn.Linear(feature_dim, hidden_dim)
        self.wavelet_low_proj = nn.Linear(feature_dim, hidden_dim)
        self.hht_low_proj = nn.Linear(feature_dim, hidden_dim)
        
        # 输出投影，将处理后的特征映射回原始维度
        self.high_output_proj = nn.Linear(hidden_dim, feature_dim)
        self.low_output_proj = nn.Linear(hidden_dim, feature_dim)
        
        if fusion_type == 'attention':
            # Cross-attention fusion
            self.query_proj = nn.Linear(hidden_dim, hidden_dim)
            self.key_proj = nn.Linear(hidden_dim, hidden_dim)
            self.value_proj = nn.Linear(hidden_dim, hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, hidden_dim)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.scale = hidden_dim ** -0.5
        
        elif fusion_type == 'tensor':
            # Tensor product fusion
            self.high_freq_mixer = nn.Sequential(
                nn.Linear(2*hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.low_freq_mixer = nn.Sequential(
                nn.Linear(2*hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
    
    def attention_fusion(self, wavelet_comp, hht_comp):
        """
        Fuse components using cross-attention with dimension adjustment
        """
        # 首先将输入投影到统一的隐藏维度
        wavelet_proj = self.wavelet_high_proj(wavelet_comp)
        hht_proj = self.hht_high_proj(hht_comp)
        
        batch_size, seq_len, _ = wavelet_proj.shape
        
        # 自注意力处理
        q_wav = self.query_proj(wavelet_proj)
        k_wav = self.key_proj(wavelet_proj)
        v_wav = self.value_proj(wavelet_proj)
        
        q_hht = self.query_proj(hht_proj)
        k_hht = self.key_proj(hht_proj)
        v_hht = self.value_proj(hht_proj)
        
        # 计算交叉注意力分数
        scores_wav_hht = torch.matmul(q_wav, k_hht.transpose(-2, -1)) * self.scale
        attn_wav_hht = F.softmax(scores_wav_hht, dim=-1)
        
        scores_hht_wav = torch.matmul(q_hht, k_wav.transpose(-2, -1)) * self.scale
        attn_hht_wav = F.softmax(scores_hht_wav, dim=-1)
        
        # 应用注意力
        output_wav = torch.matmul(attn_wav_hht, v_hht)
        output_hht = torch.matmul(attn_hht_wav, v_wav)
        
        # 残差连接 - 现在都在统一的隐藏维度中
        output_wav = self.norm1(output_wav + wavelet_proj)
        output_hht = self.norm2(output_hht + hht_proj)
        
        # 最终融合，使用可学习权重
        w_norm = torch.abs(self.wavelet_weight_high) / (torch.abs(self.wavelet_weight_high) + torch.abs(self.hht_weight_high) + 1e-8)
        h_norm = torch.abs(self.hht_weight_high) / (torch.abs(self.wavelet_weight_high) + torch.abs(self.hht_weight_high) + 1e-8)
        
        # 融合后映射回原始特征维度
        fused = w_norm * output_wav + h_norm * output_hht
        return self.high_output_proj(fused)
    
    def tensor_fusion(self, wavelet_comp, hht_comp, is_high_freq=True):
        """
        Fuse components using tensor product with dimension adjustment
        """
        # 首先将输入投影到统一的隐藏维度
        if is_high_freq:
            wavelet_proj = self.wavelet_high_proj(wavelet_comp)
            hht_proj = self.hht_high_proj(hht_comp)
        else:
            wavelet_proj = self.wavelet_low_proj(wavelet_comp)
            hht_proj = self.hht_low_proj(hht_comp)
        
        # 连接特征
        fusion_input = torch.cat([wavelet_proj, hht_proj], dim=-1)
        
        # 应用适当的混合网络
        if is_high_freq:
            fused = self.high_freq_mixer(fusion_input)
            return self.high_output_proj(fused)
        else:
            fused = self.low_freq_mixer(fusion_input)
            return self.low_output_proj(fused)
    
    def weighted_fusion(self, wavelet_comp, hht_comp, wavelet_weight, hht_weight, is_high_freq=True):
        """
        Fuse components using weighted average with dimension adjustment
        """
        # 首先将输入投影到统一的隐藏维度
        if is_high_freq:
            wavelet_proj = self.wavelet_high_proj(wavelet_comp)
            hht_proj = self.hht_high_proj(hht_comp)
            output_proj = self.high_output_proj
        else:
            wavelet_proj = self.wavelet_low_proj(wavelet_comp)
            hht_proj = self.hht_low_proj(hht_comp)
            output_proj = self.low_output_proj
        
        # 标准化权重
        w_norm = torch.abs(wavelet_weight) / (torch.abs(wavelet_weight) + torch.abs(hht_weight) + 1e-8)
        h_norm = torch.abs(hht_weight) / (torch.abs(wavelet_weight) + torch.abs(hht_weight) + 1e-8)
        
        # 加权融合在隐藏维度中进行
        fused = w_norm * wavelet_proj + h_norm * hht_proj
        
        # 映射回原始特征维度
        return output_proj(fused)
    
    def compute_energy_consistency(self, original, high_freq, low_freq):
        """
        Compute energy consistency loss
        """
        # Energy of original signal
        original_energy = torch.mean(original**2)
        
        # Energy of components
        component_energy = torch.mean(high_freq**2) + torch.mean(low_freq**2)
        
        # Energy should be preserved
        return F.mse_loss(original_energy, component_energy)
    
    def forward(self, wavelet_out, hht_out, original_signal=None):
        """
        Fuse wavelet and HHT decomposition results with length adjustment
        """
        # 首先确保长度一致
        wav_high = wavelet_out['high_freq']
        hht_high = hht_out['high_freq']
        wav_low = wavelet_out['low_freq']
        hht_low = hht_out['low_freq']
        
        # 获取各组件的形状
        b_wav_high, l_wav_high, f_wav_high = wav_high.shape
        b_hht_high, l_hht_high, f_hht_high = hht_high.shape
        b_wav_low, l_wav_low, f_wav_low = wav_low.shape
        b_hht_low, l_hht_low, f_hht_low = hht_low.shape
        
        # 调整序列长度
        min_high_len = min(l_wav_high, l_hht_high)
        min_low_len = min(l_wav_low, l_hht_low)
        
        wav_high = wav_high[:, :min_high_len, :]
        hht_high = hht_high[:, :min_high_len, :]
        wav_low = wav_low[:, :min_low_len, :]
        hht_low = hht_low[:, :min_low_len, :]
        
        # High frequency fusion
        if self.fusion_type == 'attention':
            high_freq = self.attention_fusion(wav_high, hht_high)
            low_freq = self.attention_fusion(wav_low, hht_low)
        elif self.fusion_type == 'tensor':
            high_freq = self.tensor_fusion(wav_high, hht_high, is_high_freq=True)
            low_freq = self.tensor_fusion(wav_low, hht_low, is_high_freq=False)
        else:  # Default: weighted
            high_freq = self.weighted_fusion(
                wav_high, hht_high,
                self.wavelet_weight_high, self.hht_weight_high,
                is_high_freq=True
            )
            low_freq = self.weighted_fusion(
                wav_low, hht_low,
                self.wavelet_weight_low, self.hht_weight_low,
                is_high_freq=False
            )
        
        # 调整原始信号的长度以匹配组件
        if original_signal is not None:
            original_signal = original_signal[:, :min_high_len, :]
            # Compute energy consistency
            energy_loss = self.compute_energy_consistency(
                original_signal, high_freq, low_freq[:, :min_high_len, :]
            )
        else:
            energy_loss = None
        
        return {
            'high_freq': high_freq,
            'low_freq': low_freq,
            'energy_loss': energy_loss,
            'wavelet_weight_high': self.wavelet_weight_high.item(),
            'hht_weight_high': self.hht_weight_high.item(),
            'wavelet_weight_low': self.wavelet_weight_low.item(),
            'hht_weight_low': self.hht_weight_low.item()
        }