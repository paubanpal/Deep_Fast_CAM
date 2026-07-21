import torch
import torch.nn as nn

# 1. Define the Temporal Transformer Encoder architecture
class SmallTemporalTransformer(nn.Module):
    def __init__(self, spatial_dim=128, latent_dim=128, nhead=4, num_layers=2, num_coefficients=3):
        """
        num_coefficients: output dimension. Should be de # of modes to calculate
        """
        super().__init__()
        
        # Since spatial_dim and latent_dim are equal (128), this projection maintains the size
        # but helps the network have an initial linear adaptation layer.
        self.input_projection = nn.Linear(spatial_dim, latent_dim)
        
        # Positional embedding so the Transformer knows the order of the 500 frames
        self.positional_embedding = nn.Parameter(torch.randn(1, 500, latent_dim))
        
        # The core of the model: A Pure Encoder (Full bidirectional attention, no mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, 
            nhead=nhead, 
            dim_feedforward=latent_dim * 2, 
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Regression head to get the distortion coefficients for each frame
        self.coefficient_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, num_coefficients)
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Projection and time injection
        x = self.input_projection(x)
        x = x + self.positional_embedding[:, :seq_len, :]
        
        # The Transformer analyzes the entire block forward and backward at the same time
        x = self.transformer(x)
        
        # Final mapping to coefficients
        coefficients = self.coefficient_head(x)
        return coefficients

# =====================================================================
# 2. DATA FLOW SIMULATION (USAGE EXAMPLE)
# =====================================================================
# Instantiate the model with your optimal dimensions
# We assume you need to predict 3 distortion coefficients per image (e.g., scale, X translation, Y translation)
temporal_model = SmallTemporalTransformer(
    spatial_dim=128, 
    latent_dim=128, 
    nhead=4, 
    num_layers=2, 
    num_coefficients=3
)

# Simulate processing a "Batch" of 2 blocks at a time.
# Each block contains 500 images, and your first network has already converted them into vectors of size 128.
# Expected shape: [Batch_Size, Number_of_Frames, Spatial_Latent_Dimension]
block_latent_features = torch.randn(2, 500, 128)

print("--- TENSOR FLOW ---")
print(f"1. Input coming from the spatial network: {block_latent_features.shape}")

# Pass the data through the Temporal Transformer
predicted_coefficients = temporal_model(block_latent_features)

print(f"2. Transformer output (Coefficients): {predicted_coefficients.shape}")

# =====================================================================
# 3. HOW DOES THIS CONNECT TO YOUR LOSS FUNCTION?
# =====================================================================
print("\n--- POST-STEP (LOSS FUNCTION) ---")
# Extract the coefficients of the first block in the batch to see what it predicted
block_1_coefficients = predicted_coefficients[0] # Shape: [500, 3]

print(f"For Frame 1 of Block 1, the coefficients are: {block_1_coefficients[0].detach().numpy()}")
print(f"For Frame 500 of Block 1, the coefficients are: {block_1_coefficients[499].detach().numpy()}")

print("\n> Now these 500 sets of coefficients would go to your physics-based loss function")
print("> along with the original 128x128 images to evaluate the underlying reconstruction.")