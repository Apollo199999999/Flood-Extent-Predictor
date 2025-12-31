import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Patch Embedding
class PatchEmbedding(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=500, patch_size=10, in_channels=5, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # A convolution layer that splits the image into patches and embeds them.
        # The kernel size and stride should be equal to the patch size.
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, C, H, W) -> (B, embed_dim, H'/patch_size, W'/patch_size)
        # Flatten the feature map and rearrange dimensions to (B, num_patches, embed_dim).
        x = x.flatten(2).transpose(1, 2)
        return x
    
# Token mixing in ViT to add local features
class LocalMixer(nn.Module):
    # ResNet inspired

    def __init__(self, embed_dim, image_size, patch_size, dropout=0.1):
        super().__init__()
        self.h, self.w = image_size // patch_size, image_size // patch_size

        # To learn features between tokens
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)

        self.gelu = nn.GELU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: (N, num_patches, embed_dim)
        N, num_patches, embed_dim = x.shape
        x2 = x.transpose(1,2).reshape(N, embed_dim, self.h, self.w)
        res = x2

        # Apply the conv operations
        x2 = self.conv1(res)
        x2 = self.gelu(x2)
        x2 = self.conv2(x2)
        x2 = self.gelu(x2)

        # Residual connection
        x2 = x2 + self.dropout(res)

        x2 = x2.flatten(2).transpose(1, 2)

        return x2


# 1D sinusoidal positional embeddings
class PositionalEmbedding1D(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()

        # Positional embeddings give the model information about the patch order.
        # Use sinusoidal embeddings, as is the convention
        pe = torch.zeros(num_patches, embed_dim)
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe

# 2D sinusoidal positional embeddings
# The idea is for a vector with K dimensions, the first K / 2 dimensions encodes X positional info, while the remaining K / 2 encodes Y positional info
# We then do this for all vectors representing each patch
# This code was mostly adapted from https://github.com/s-chh/2D-Positional-Encoding-Vision-Transformer
# In the comments below, N refers to number of patches, not batch size
class PositionalEmbedding2D(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        
        # Split the dimensions into two parts. We will use 1 part for x-axis and the other part for y-axis
        self.embed_dim  = embed_dim // 2                                             

        # X-axis specific values
        x_positions = self.get_x_positions(num_patches).reshape(-1, 1) # N  ->  N, 1
        x_pos_embedding = self.generate_sinusoidal1D(x_positions) # 1, N, E // 2

        # Y-axis specific values
        y_positions = self.get_y_positions(num_patches).reshape(-1, 1) # N  ->  N, 1
        y_pos_embedding = self.generate_sinusoidal1D(y_positions) # 1, N, E//2

        # Combine x-axis and y-axis positional encodings
        pos_embedding = torch.cat((x_pos_embedding, y_pos_embedding), -1)  # 1, N, E//2  concat  1, N, E//2  ->  1, N, E
        self.register_buffer("pe", pos_embedding) # Register_buffer for easy switching of device

    # X-axis specific values
    def get_x_positions(self, num_patches, start_idx=0):
        n_patches_ = int(num_patches ** 0.5)  # Number of patches along 1 dimension

        x_positions = torch.arange(start_idx, n_patches_ + start_idx)         # N_
        x_positions = x_positions.unsqueeze(0)                                # 1, N_
        x_positions = torch.repeat_interleave(x_positions, n_patches_, 0)     # N_ , N_ Matrix to replicate positions of patches on x-axis
        x_positions = x_positions.reshape(-1)                                 # N_ , N_  ->  N_ ** 2  =  N

        return x_positions

    # Y-axis specific values
    def get_y_positions(self, num_patches, start_idx=0):
        n_patches_ = int(num_patches ** 0.5)                                   # Number of patches along 1 dimension

        y_positions = torch.arange(start_idx, n_patches_+start_idx)           # N_
        y_positions = torch.repeat_interleave(y_positions, n_patches_, 0)     # N_ , N_  ->  N_ ** 2  =  N  Matrix to replicate positions of patches on y-axis

        return y_positions

    def generate_sinusoidal1D(self, sequence):
        # Denominator
        denominator = torch.pow(10000, torch.arange(0, self.embed_dim, 2) / self.embed_dim)   # E//4  Denominator used to produce sinusoidal equation

        # Create an empty tensor and fill with sin and cos values as per sinusoidal embedding equation
        pos_embedding = torch.zeros(1, sequence.shape[0], self.embed_dim)                     # 1, N, E//2  Used to store positional embedding for x/y-axis variations
        denominator = sequence / denominator                                                  # N, 1 / (E//4)  ->  N, E//4
        pos_embedding[:, :, ::2]  = torch.sin(denominator)                                    # Fill positional embedding's even dimensions with sin values
        pos_embedding[:, :, 1::2] = torch.cos(denominator)                                    # Fill positional embedding's odd dimensions with cos values
        return pos_embedding                                                

    def forward(self, x):
        return x + self.pe


# A Mixture of Experts module to replace the normal MLP in ViTs
class Expert(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        ) 
 
    def forward(self, x):
        return self.mlp(x)
 
class MoELayer(nn.Module):
    def __init__(self, embed_dim, mlp_dim, num_experts=4, top_k=1, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = embed_dim
        # Create expert networks
        self.experts = nn.ModuleList([
            Expert(embed_dim, mlp_dim, dropout) for _ in range(num_experts)
        ])
        self.router = nn.Linear(embed_dim, num_experts)

        # Use loss-free load balancing as outlined in the DeepSeek paper pg 4 Algorithm 1: https://arxiv.org/pdf/2408.15664
        self.register_buffer("expert_bias", torch.zeros(num_experts))
 
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Reshape for expert processing, the compute routing probabilities
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        # Add the expert biases to the gating network probabilites for load balancing
        router_logits = self.router(hidden_states_reshaped) + self.expert_bias.view(1, -1)  # (batch_size * seq_len, num_experts)
        routing_probs = F.softmax(router_logits, dim=-1)
 
        # Select top-k experts, and scale the probabilities to sum to 1
        # output shape: (batch_size * seq_len, k)
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Loss free load balancing: https://arxiv.org/pdf/2408.15664
        with torch.no_grad():
            # Initialise an array to count the number of tokens per expert, so we can update expert biases
            expert_num_tokens = torch.zeros_like(self.expert_bias)

            for i in range(self.top_k):
                expert_num_tokens.scatter_add_(
                    0,
                    top_k_indices[:, i],
                    torch.ones_like(top_k_probs[:, i])
                )

            mean_num_tokens = expert_num_tokens.mean()
            load_error = mean_num_tokens - expert_num_tokens
            # In-place update (keeps this as a proper buffer)
            self.expert_bias.add_(0.001 * torch.sign(load_error))

        # Process through selected experts
        expert_output = torch.zeros_like(hidden_states_reshaped)

        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i] # (batch_size * seq_len)
            expert_probs = top_k_probs[:, i] # (batch_size * seq_len)
            
            for j in range(self.num_experts):
                # Route the necessary tokens to each expert
                mask = (expert_idx == j)
                if not mask.any():
                    continue
                expert_tokens = hidden_states_reshaped[mask]
                expert_output[mask] = expert_output[mask] + (expert_probs[mask].view(-1, 1) * self.experts[j](expert_tokens))

        # Reshape back to original shape
        output = expert_output.view(batch_size, seq_len, hidden_dim)
        return output

# The normal transformer encoder for ViTs
class BaseTransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        # Use nn.MultiheadAttention. Ensure batch_first=True.
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(embed_dim)
        # A simple MLP with two linear layers and a GELU activation in between.
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Attention block with residual connection
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        # Add the output of the attention layer to the original input (skip connection).
        x = x + attn_output

        # MLP block with residual connection
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        # Add the output of the MLP to the result of the first skip connection.
        x = x + mlp_output
        return x
    
# The improved transformer encoder for flood detection, with Mixture of Experts and Token Mixing
class FloodTransformerEncoder(nn.Module):
    def __init__(self, img_size=500, patch_size=10, embed_dim=768, num_heads=8, mlp_dim=2048, 
                 num_experts=4, num_shared_experts=1, top_k=1, dropout=0.1):
        super().__init__()

        # Use nn.MultiheadAttention. Ensure batch_first=True.
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Token mixing
        self.mixer = LocalMixer(embed_dim, img_size, patch_size, dropout)  

        # Norm layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Replace MLP with MoE layer
        self.moe = MoELayer(embed_dim, mlp_dim, num_experts, top_k, dropout)

        # Shared experts, which is what DeepSeek uses
        self.shared_experts = nn.ModuleList([
            Expert(embed_dim, mlp_dim, dropout) for _ in range(num_shared_experts)
        ])

    def forward(self, x):
        # Attention block with residual connection
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)

        # Add the output of the attention layer to the original input (skip connection).
        x = x + attn_output

        # Before MLP, mix tokens
        x = self.mixer(x)

        # MoE block with residual connection
        # MoE sublayer
        x = self.norm2(x)

        moe_output = self.moe(x)

        for expert in self.shared_experts:
            moe_output = moe_output + expert(x)

        # Residual connection
        return x + moe_output

# The classic vision transformer, with litle architectural changes
class BaseViT(nn.Module):
    def __init__(self, img_size=500, patch_size=10, in_channels=5,
                 embed_dim=768, depth=8, num_heads=8, mlp_dim=2048, dropout=0.1):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        # Instantiate the PatchEmbedding module.
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Positional embeddings 
        self.pos_embed = PositionalEmbedding1D(num_patches, embed_dim)

        # Create a sequence of BaseTransformerEncoder blocks.
        self.blocks = nn.ModuleList([
            BaseTransformerEncoder(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        B = x.shape[0]

        # Implement the forward pass in the ViT
        # 1. Get patch embeddings from x.
        # 2. Add positional embeddings.
        # 3. Pass the result through the Transformer encoder blocks.
        # 4. Apply layer norm, and pass it to the classification head.

        x = self.patch_embed(x)
        x = self.pos_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.head(x)
        x = x.view(-1, self.img_size // self.patch_size, self.img_size // self.patch_size)

        return x

# The improved vision transformer, with architectural changes to hopefully help with flood detection
class FloodViT(nn.Module):
    def __init__(self, img_size=500, patch_size=10, in_channels=5,
                 embed_dim=768, depth=8, num_heads=8, mlp_dim=2048, 
                 num_experts=4, num_shared_experts=1, top_k=1, dropout=0.1):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        # Instantiate the PatchEmbedding module.
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 2D Positional embeddings 
        self.pos_embed = PositionalEmbedding2D(num_patches, embed_dim)

        # Create a sequence of TransformerEncoder blocks.
        self.blocks = nn.ModuleList([
            FloodTransformerEncoder(img_size, 
                                    patch_size, 
                                    embed_dim, 
                                    num_heads, 
                                    mlp_dim, 
                                    num_experts,
                                    num_shared_experts, 
                                    top_k,
                                    dropout) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        B = x.shape[0]

        # Implement the forward pass in the ViT
        # 1. Get patch embeddings from x.
        # 2. Add positional embeddings.
        # 3. Pass the result through the Transformer encoder blocks.
        # 4. Apply layer norm, and pass it to the classification head.

        x = self.patch_embed(x)
        x = self.pos_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.head(x)
        x = x.view(-1, self.img_size // self.patch_size, self.img_size // self.patch_size)

        return x