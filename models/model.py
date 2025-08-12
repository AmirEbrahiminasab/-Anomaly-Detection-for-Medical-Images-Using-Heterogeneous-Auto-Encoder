# main_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Dict


# --- 1. Encoder Implementation ---
# The encoder uses a pre-trained ResNet to extract hierarchical features.
# [cite_start]Its weights are frozen during training. [cite: 235, 189]

class Encoder(nn.Module):
    """
    Encoder module using a pre-trained ResNet-18.
    It extracts features from the first four stages.
    """

    def __init__(self):
        super(Encoder, self).__init__()
        # [cite_start]Load a pre-trained ResNet-18 model [cite: 235]
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # We extract features from these layers
        self.stage1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.stage2 = resnet.layer1
        self.stage3 = resnet.layer2
        self.stage4 = resnet.layer3  # The paper uses the first four stages, in ResNet-18, layer4 is the 5th stage.

        # [cite_start]Freeze the encoder weights as it's not trained [cite: 189]
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass to extract hierarchical features.
        [cite_start]Returns a dictionary of feature maps from each stage. [cite: 238]
        """
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        return {'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4}


# --- 2. Decoder Implementation (Hybrid CNN-Transformer Network) ---
# This is the core of the novel architecture, designed to have different
# [cite_start]inductive biases than the CNN-based encoder. [cite: 226]

# --- 2a. Multi-scale Sparse Transformer Block (MSTB) ---
class MSTB(nn.Module):
    """
    [cite_start]Multi-scale Sparse Transformer Block. [cite: 255]
    Models relationships between local queries and multi-scale regional keys/values
    [cite_start]to balance performance and computational cost. [cite: 260, 274, 81]
    """

    def __init__(self, in_channels: int, num_heads: int, p1_size: int = 4, p2_size: int = 8):
        super(MSTB, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K, V
        self.to_q = nn.Linear(in_channels, in_channels)
        self.to_k = nn.Linear(in_channels, in_channels)
        self.to_v = nn.Linear(in_channels, in_channels)
        self.to_out = nn.Linear(in_channels, in_channels)

        # Patch embedding for regional information
        self.patch_embed_p1 = nn.Conv2d(in_channels, in_channels, kernel_size=p1_size, stride=p1_size)
        self.patch_embed_p2 = nn.Conv2d(in_channels, in_channels, kernel_size=p2_size, stride=p2_size)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 4 * in_channels),
            nn.GELU(),
            nn.Linear(4 * in_channels, in_channels)
        )
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # [cite_start]Local Information (Query) [cite: 264]
        q_local = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        q = self.to_q(self.norm1(q_local)).reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # [cite_start]Regional Information (Key, Value) at two scales [cite: 270]
        # Scale 1
        x_p1 = self.patch_embed_p1(x).flatten(2).transpose(1, 2)  # (B, Np1, C)
        k_p1 = self.to_k(x_p1).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_p1 = self.to_v(x_p1).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scale 2
        x_p2 = self.patch_embed_p2(x).flatten(2).transpose(1, 2)  # (B, Np2, C)
        k_p2 = self.to_k(x_p2).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_p2 = self.to_v(x_p2).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # [cite_start]Multi-head Attention with multi-scale regional info [cite: 286]
        # Split heads to process different scales
        q1, q2 = torch.chunk(q, 2, dim=1)  # Split heads for multi-scale attention

        attn_p1 = (q1 @ k_p1.transpose(-2, -1)) * self.scale
        attn_p1 = attn_p1.softmax(dim=-1)
        out_p1 = (attn_p1 @ v_p1).transpose(1, 2).reshape(B, H * W, C // 2)

        attn_p2 = (q2 @ k_p2.transpose(-2, -1)) * self.scale
        attn_p2 = attn_p2.softmax(dim=-1)
        out_p2 = (attn_p2 @ v_p2).transpose(1, 2).reshape(B, H * W, C // 2)

        # Concatenate features from both scales
        out = torch.cat([out_p1, out_p2], dim=-1)
        out = self.to_out(out)

        # FFN
        out = out + q_local
        out = self.mlp(self.norm2(out)) + out

        return out.transpose(1, 2).reshape(B, C, H, W)


# --- 2b. Hybrid CNN-Transformer Module ---
class HybridModule(nn.Module):
    """
    [cite_start]The hybrid module containing parallel CNN and Transformer blocks. [cite: 245]
    """

    def __init__(self, in_channels: int, out_channels: int, num_heads: int, p_sizes: tuple):
        super(HybridModule, self).__init__()

        # [cite_start]Split input into two paths [cite: 246]
        self.conv1_1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv1_2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

        # [cite_start]Convolutional Block [cite: 255]
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1)
        )

        # Transformer Block (MSTB)
        self.transformer_block = MSTB(in_channels // 2, num_heads, p1_size=p_sizes[0], p2_size=p_sizes[1])

        # [cite_start]Concatenation and Upsampling [cite: 249, 250]
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input for parallel processing
        x_conv = self.conv1_1(x)
        x_trans = self.conv1_2(x)

        # Process through CNN and Transformer paths
        f_conv = self.conv_block(x_conv)  # [cite: 247]
        f_trans = self.transformer_block(x_trans)  # [cite: 248]

        # Concatenate and process
        f_cat = torch.cat([f_conv, f_trans], dim=1)  # [cite: 249]
        f_out = self.conv_out(f_cat)

        return self.upsample(f_out)


# --- 2c. Full Decoder ---
class Decoder(nn.Module):
    """
    The full decoder, stacking multiple HybridModules to reconstruct features.
    [cite_start]It takes the most compressed feature F_E^4 as input. [cite: 241]
    """

    def __init__(self):
        super(Decoder, self).__init__()
        # Decoder stages are designed to match encoder output dimensions
        # Stage 3 reconstructs F_D^3 from F_E^4 to match F_E^3
        self.stage3 = HybridModule(in_channels=256, out_channels=128, num_heads=8, p_sizes=(2, 4))
        # Stage 2 reconstructs F_D^2 to match F_E^2
        self.stage2 = HybridModule(in_channels=128, out_channels=64, num_heads=4, p_sizes=(4, 8))
        # Stage 1 reconstructs F_D^1 to match F_E^1
        self.stage1 = HybridModule(in_channels=64, out_channels=64, num_heads=2, p_sizes=(8, 16))

    def forward(self, f4: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        [cite_start]Forward pass to reconstruct hierarchical features. [cite: 243]
        """
        f3_dec = self.stage3(f4)
        f2_dec = self.stage2(f3_dec)
        f1_dec = self.stage1(f2_dec)

        return {'f1': f1_dec, 'f2': f2_dec, 'f3': f3_dec}


# --- 3. Full Hetero-AE Model ---
class HeteroAE(nn.Module):
    """
    The complete Heterogeneous Auto-Encoder model.
    """

    def __init__(self):
        super(HeteroAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x: torch.Tensor) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        encoder_features = self.encoder(x)
        decoder_features = self.decoder(encoder_features['f4'])
        return encoder_features, decoder_features


# --- 4. Loss Function and Anomaly Scoring ---

class FeatureComparisonLoss(nn.Module):
    """
    [cite_start]Multi-stage feature comparison loss, combining Cosine Similarity and MSE. [cite: 290]
    """

    def __init__(self, alpha: float = 0.7):
        super(FeatureComparisonLoss, self).__init__()
        self.alpha = alpha

    def forward(self, features_enc: Dict[str, torch.Tensor], features_dec: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = 0.0
        num_stages = 0

        # [cite_start]Iterate over decoder stages (1, 2, 3) [cite: 294]
        for key in features_dec:
            f_enc = features_enc[key]
            f_dec = features_dec[key]

            # [cite_start]Calculate Cosine Similarity Loss and MSE Loss [cite: 291]
            cos_loss = 1 - F.cosine_similarity(f_enc, f_dec, dim=1).mean()
            mse_loss = F.mse_loss(f_enc, f_dec)

            stage_loss = self.alpha * cos_loss + (1 - self.alpha) * mse_loss
            total_loss += stage_loss
            num_stages += 1

        return total_loss / num_stages


def calculate_anomaly_map(features_enc: Dict[str, torch.Tensor], features_dec: Dict[str, torch.Tensor],
                          input_size: tuple = (256, 256)) -> tuple[torch.Tensor, torch.Tensor]:
    """
    [cite_start]Calculates the final anomaly map and score. [cite: 334]
    """
    anomaly_map = torch.zeros(features_enc['f1'].shape[0], 1, input_size[0], input_size[1]).to(
        features_enc['f1'].device)

    for key in features_dec:
        f_enc = features_enc[key]
        f_dec = features_dec[key]

        # [cite_start]Calculate stage-wise anomaly map M_k [cite: 337]
        m_k = 1 - F.cosine_similarity(f_enc, f_dec, dim=1).unsqueeze(1)

        # [cite_start]Resize to input image resolution and accumulate [cite: 335]
        m_k_resized = F.interpolate(m_k, size=input_size, mode='bilinear', align_corners=False)
        anomaly_map += m_k_resized

    # [cite_start]Anomaly score is the max value in the map [cite: 343]
    anomaly_score = torch.max(anomaly_map.flatten(1), dim=1)[0]

    return anomaly_map, anomaly_score


# --- 5. Example Usage ---
if __name__ == '__main__':
    # --- Model Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeteroAE().to(device)
    loss_fn = FeatureComparisonLoss(alpha=0.7).to(device)

    # [cite_start]In training, only decoder parameters are optimized [cite: 189]
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=1e-4)
    print(f"Model, Encoder, and Decoder initialized on {device}.")

    # --- Mock Training Loop ---
    print("\n--- Mock Training ---")
    model.train()
    # Create a dummy batch of normal images
    dummy_normal_images = torch.randn(4, 3, 256, 256).to(device)

    optimizer.zero_grad()
    encoder_features, decoder_features = model(dummy_normal_images)
    loss = loss_fn(encoder_features, decoder_features)
    loss.backward()
    optimizer.step()

    print(f"Training step completed. Loss: {loss.item():.4f}")

    # --- Mock Inference (Testing) Loop ---
    print("\n--- Mock Inference ---")
    model.eval()
    # Create a dummy test image (could be normal or abnormal)
    dummy_test_image = torch.randn(1, 3, 256, 256).to(device)

    with torch.no_grad():
        enc_feats, dec_feats = model(dummy_test_image)
        anomaly_map, anomaly_score = calculate_anomaly_map(enc_feats, dec_feats, input_size=(256, 256))

    print(f"Inference step completed.")
    print(f"Anomaly Map Shape: {anomaly_map.shape}")
    print(f"Anomaly Score: {anomaly_score.item():.4f}")

    # --- Parameter Count ---
    num_encoder_params = sum(p.numel() for p in model.encoder.parameters())
    num_decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print("\n--- Parameter Counts ---")
    print(f"Encoder (frozen) parameters: {num_encoder_params / 1e6:.2f}M")
    print(f"Decoder (trainable) parameters: {num_decoder_params / 1e6:.2f}M")

