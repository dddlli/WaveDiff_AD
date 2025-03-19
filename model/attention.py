import numpy as np
import torch
from torch import nn


class OrdinaryAttention(nn.Module):
    """
    Standard multi-head self-attention mechanism
    
    Args:
        model_dim: Dimension of model features
        atten_dim: Dimension of attention heads
        head_num: Number of attention heads
        dropout: Dropout rate
        residual: Whether to use residual connection
    """
    def __init__(self, model_dim, atten_dim, head_num, dropout, residual=True):
        super(OrdinaryAttention, self).__init__()
        self.atten_dim = atten_dim
        self.head_num = head_num
        self.residual = residual

        # Projections for query, key, and value
        self.W_Q = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_K = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_V = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)

        # Output projection
        self.fc = nn.Linear(self.atten_dim * self.head_num, model_dim, bias=True)

        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, Q, K=None, V=None):
        """
        Forward pass of self-attention
        
        Args:
            Q: Query tensor (batch, seq_len, dim)
            K: Key tensor (uses Q if None)
            V: Value tensor (uses Q if None)
            
        Returns:
            Attention output
        """
        if K is None:
            K = Q
        if V is None:
            V = Q
            
        batch_size, q_len, _ = Q.shape
        k_len = K.shape[1]
        
        residual = Q.clone()

        # Linear projections and reshape for multi-head
        Q = self.W_Q(Q).view(batch_size, q_len, self.head_num, self.atten_dim)
        K = self.W_K(K).view(batch_size, k_len, self.head_num, self.atten_dim)
        V = self.W_V(V).view(batch_size, k_len, self.head_num, self.atten_dim)

        # Transpose for attention computation [batch, head, seq_len, dim]
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        attn = nn.Softmax(dim=-1)(scores)
        
        # Apply attention to values
        context = torch.matmul(attn, V)

        # Reshape and project back
        context = context.transpose(1, 2)  # [batch, seq_len, heads, dim]
        context = context.reshape(batch_size, q_len, -1)  # [batch, seq_len, heads*dim]
        output = self.dropout(self.fc(context))

        # Apply residual connection if specified
        if self.residual:
            return self.norm(output + residual)
        else:
            return self.norm(output)


class MixAttention(nn.Module):
    """
    Mixed attention for combining information from different modalities
    
    Args:
        model_dim: Dimension of model features
        atten_dim: Dimension of attention heads
        head_num: Number of attention heads
        dropout: Dropout rate
        residual: Whether to use residual connection
    """
    def __init__(self, model_dim, atten_dim, head_num, dropout, residual=True):
        super(MixAttention, self).__init__()
        self.atten_dim = atten_dim
        self.head_num = head_num
        self.residual = residual

        # Projections for query, key, and value - separate for data and time
        self.W_Q_data = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_Q_time = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_K_data = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_K_time = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_V_data = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_V_time = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)

        # Output projection
        self.fc = nn.Linear(self.atten_dim * self.head_num, model_dim, bias=True)

        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, Q_data, Q_time, K_data, K_time, V_data, V_time):
        """
        Forward pass of mixed attention
        
        Args:
            Q_data: Data query tensor
            Q_time: Time query tensor
            K_data: Data key tensor
            K_time: Time key tensor
            V_data: Data value tensor
            V_time: Time value tensor
            
        Returns:
            Mixed attention output
        """
        residual = Q_data.clone()
        batch_size, seq_len, _ = Q_data.shape

        # Linear projections and reshape for multi-head
        Q_data = self.W_Q_data(Q_data).view(batch_size, seq_len, self.head_num, self.atten_dim)
        Q_time = self.W_Q_time(Q_time).view(batch_size, seq_len, self.head_num, self.atten_dim)
        K_data = self.W_K_data(K_data).view(batch_size, seq_len, self.head_num, self.atten_dim)
        K_time = self.W_K_time(K_time).view(batch_size, seq_len, self.head_num, self.atten_dim)
        V_data = self.W_V_data(V_data).view(batch_size, seq_len, self.head_num, self.atten_dim)
        V_time = self.W_V_time(V_time).view(batch_size, seq_len, self.head_num, self.atten_dim)

        # Transpose for attention computation [batch, head, seq_len, dim]
        Q_data, Q_time = Q_data.transpose(1, 2), Q_time.transpose(1, 2)
        K_data, K_time = K_data.transpose(1, 2), K_time.transpose(1, 2)
        V_data, V_time = V_data.transpose(1, 2), V_time.transpose(1, 2)

        # Compute attention scores - mix data and time influences
        scores_data = torch.matmul(Q_data, K_data.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        scores_time = torch.matmul(Q_time, K_time.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        
        # Combine scores and apply softmax
        scores_combined = scores_data + scores_time
        attn = nn.Softmax(dim=-1)(scores_combined)
        
        # Combine values from data and time
        context_data = torch.matmul(attn, V_data)
        context_time = torch.matmul(attn, V_time)
        context = context_data + context_time

        # Reshape and project back
        context = context.transpose(1, 2)  # [batch, seq_len, heads, dim]
        context = context.reshape(batch_size, seq_len, -1)  # [batch, seq_len, heads*dim]
        output = self.dropout(self.fc(context))

        # Apply residual connection if specified
        if self.residual:
            return self.norm(output + residual)
        else:
            return self.norm(output)


class TimeAwareAttention(nn.Module):
    """
    Time-aware attention for temporal relationships
    
    Args:
        model_dim: Dimension of model features
        atten_dim: Dimension of attention heads
        head_num: Number of attention heads
        max_len: Maximum sequence length
        dropout: Dropout rate
        residual: Whether to use residual connection
    """
    def __init__(self, model_dim, atten_dim, head_num, max_len=5000, dropout=0.1, residual=True):
        super(TimeAwareAttention, self).__init__()
        self.atten_dim = atten_dim
        self.head_num = head_num
        self.max_len = max_len
        self.residual = residual
        
        # Standard attention projections
        self.W_Q = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_K = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_V = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        
        # Time-aware projections
        self.time_relation = nn.Parameter(torch.Tensor(max_len, max_len))
        nn.init.xavier_uniform_(self.time_relation)
        self.time_proj = nn.Sequential(
            nn.Linear(1, atten_dim),
            nn.ReLU(),
            nn.Linear(atten_dim, 1)
        )
        
        # Output projection
        self.fc = nn.Linear(self.atten_dim * self.head_num, model_dim, bias=True)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)
        
    def forward(self, Q, time_indices=None, K=None, V=None):
        """
        Forward pass of time-aware attention
        
        Args:
            Q: Query tensor
            time_indices: Time indices for sequence positions
            K: Key tensor (uses Q if None)
            V: Value tensor (uses Q if None)
            
        Returns:
            Time-aware attention output
        """
        if K is None:
            K = Q
        if V is None:
            V = Q
            
        batch_size, seq_len, _ = Q.shape
        residual = Q.clone()
        
        # Create default time indices if not provided
        if time_indices is None:
            time_indices = torch.arange(seq_len, device=Q.device)
            time_indices = time_indices.unsqueeze(0).repeat(batch_size, 1)
        
        # Linear projections and reshape for multi-head
        Q = self.W_Q(Q).view(batch_size, seq_len, self.head_num, self.atten_dim)
        K = self.W_K(K).view(batch_size, seq_len, self.head_num, self.atten_dim)
        V = self.W_V(V).view(batch_size, seq_len, self.head_num, self.atten_dim)
        
        # Transpose for attention computation [batch, head, seq_len, dim]
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        
        # Extract time relations for the current sequence
        time_rel = self.time_relation[:seq_len, :seq_len]
        
        # Compute time-adjusted attention scores
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        
        # Add time relation bias to attention scores
        # First create time difference matrix
        time_indices_i = time_indices.unsqueeze(-1)  # [batch, seq_len, 1]
        time_indices_j = time_indices.unsqueeze(1)   # [batch, 1, seq_len]
        time_diff = (time_indices_i - time_indices_j).float().unsqueeze(-1)  # [batch, seq_len, seq_len, 1]
        
        # Project time differences
        time_bias = self.time_proj(time_diff).squeeze(-1)  # [batch, seq_len, seq_len]
        
        # Add time bias to scores (broadcast across heads)
        time_bias = time_bias.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
        scores = scores + time_bias + time_rel
        
        # Apply softmax
        attn = nn.Softmax(dim=-1)(scores)
        
        # Apply attention to values
        context = torch.matmul(attn, V)
        
        # Reshape and project back
        context = context.transpose(1, 2)  # [batch, seq_len, heads, dim]
        context = context.reshape(batch_size, seq_len, -1)  # [batch, seq_len, heads*dim]
        output = self.dropout(self.fc(context))
        
        # Apply residual connection if specified
        if self.residual:
            return self.norm(output + residual)
        else:
            return self.norm(output)


class ComponentAttention(nn.Module):
    """
    Attention mechanism for integrating different frequency components
    
    Args:
        model_dim: Dimension of model features
        atten_dim: Dimension of attention heads
        head_num: Number of attention heads
        dropout: Dropout rate
    """
    def __init__(self, model_dim, atten_dim, head_num, dropout=0.1):
        super(ComponentAttention, self).__init__()
        self.atten_dim = atten_dim
        self.head_num = head_num
        
        # High and low frequency projections
        self.high_freq_proj = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.low_freq_proj = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        
        # Cross-attention projections
        self.W_Q = nn.Linear(model_dim, self.atten_dim * self.head_num, bias=True)
        self.W_K = nn.Linear(self.atten_dim * self.head_num, self.atten_dim * self.head_num, bias=True)
        self.W_V = nn.Linear(self.atten_dim * self.head_num, self.atten_dim * self.head_num, bias=True)
        
        # Output projection
        self.fc = nn.Linear(self.atten_dim * self.head_num, model_dim, bias=True)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)
        
    def forward(self, x, high_freq, low_freq):
        """
        Forward pass of component attention
        
        Args:
            x: Input tensor
            high_freq: High frequency component
            low_freq: Low frequency component
            
        Returns:
            Component-integrated output
        """
        batch_size, seq_len, _ = x.shape
        residual = x.clone()
        
        # Project components
        high_feat = self.high_freq_proj(high_freq)
        low_feat = self.low_freq_proj(low_freq)
        
        # Combined component features
        comp_feat = high_feat + low_feat
        
        # Query from input, key-value from components
        Q = self.W_Q(x).view(batch_size, seq_len, self.head_num, self.atten_dim)
        K = self.W_K(comp_feat).view(batch_size, seq_len, self.head_num, self.atten_dim)
        V = self.W_V(comp_feat).view(batch_size, seq_len, self.head_num, self.atten_dim)
        
        # Transpose for attention computation [batch, head, seq_len, dim]
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.atten_dim)
        attn = nn.Softmax(dim=-1)(scores)
        
        # Apply attention to values
        context = torch.matmul(attn, V)
        
        # Reshape and project back
        context = context.transpose(1, 2)  # [batch, seq_len, heads, dim]
        context = context.reshape(batch_size, seq_len, -1)  # [batch, seq_len, heads*dim]
        output = self.dropout(self.fc(context))
        
        # Apply residual connection
        return self.norm(output + residual)