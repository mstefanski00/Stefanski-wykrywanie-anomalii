import torch
import torch.nn as nn
from typing import List

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_dims: List[int], latent_dim: int):
        super(LSTMAutoencoder, self).__init__()

        self.n_features = n_features
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        
        self.encoder_lstm = nn.LSTM(
            input_size = n_features,
            hidden_size = hidden_dims[0],
            num_layers=len(hidden_dims),
            batch_first= True,
        )

        self.encoder_projection = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], latent_dim),
        )

        self.decoder_projection = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[0]),
        )

        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dims[0],
            hidden_size=n_features,
            num_layers=len(hidden_dims),
            batch_first=True,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder_lstm(x)
        return self.encoder_projection(hidden[-1])
        
    def decode(self, z:torch.Tensor, seq_len: int) -> torch.Tensor:
        decoded = self.decoder_projection(z)
        repeated = decoded.unsqueeze(1).repeat(1, seq_len, 1)
        reconstructed, _ = self.decoder_lstm(repeated)
        return reconstructed
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z, seq_len = x.size(1))