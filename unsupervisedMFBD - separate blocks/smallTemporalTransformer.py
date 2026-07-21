import torch
import torch.nn as nn

class SmallTemporalTransformer(nn.Module):
    def __init__(self, spatial_dim=128, latent_dim=None, nhead=4, num_layers=2, num_coefficients=3, max_seq_len=1000):
        super().__init__()
        """
        num_coefficients: output dimension. Should be de # of modes to calculate
        latent_dim: if specified, it must be divisible by nhead
        """

        if latent_dim is None:
            latent_dim = spatial_dim

        if latent_dim % nhead != 0:
            raise ValueError(f"latent_dim ({latent_dim}) must be divisible by nhead ({nhead}).")
        
        # Since spatial_dim and latent_dim are equal (128), this projection maintains the size
        # but helps the network have an initial linear adaptation layer.
        self.input_projection = nn.Linear(spatial_dim, latent_dim)
        
        # Ampliamos max_seq_len (por ejemplo, a 1000) para no limitar el modelo si llega un stack grande
        self.positional_embedding = nn.Parameter(torch.randn(1, max_seq_len, latent_dim))
        
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
        

    def forward(self, x, key_padding_mask=None):
        """
        x: Tensor [Batch_Size, Seq_Len, Spatial_Dim]
        key_padding_mask: Tensor Booleano [Batch_Size, Seq_Len] 
                          (True para las posiciones de padding que DEBEN ignorarse)
        """
        batch_size, seq_len, _ = x.shape
        
        # Proyección e inyección posicional ajustada a la longitud real seq_len del lote actual
        x = self.input_projection(x)
        x = x + self.positional_embedding[:, :seq_len, :]
        
        # El Transformer ignora las posiciones marcadas como True en key_padding_mask
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        
        # Mapeo a coeficientes
        coefficients = self.coefficient_head(x)
        return coefficients
