import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, layers):
        super(Autoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        for i in range(len(layers) - 1):
            encoder_layers.append(nn.Linear(layers[i], layers[i+1]))
            # Use ReLU for intermediate layers
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (Symmetric)
        decoder_layers = []
        reversed_layers = layers[::-1]
        for i in range(len(reversed_layers) - 1):
            decoder_layers.append(nn.Linear(reversed_layers[i], reversed_layers[i+1]))
            if i == len(reversed_layers) - 2:
                # Final layer: Sigmoid (assuming inputs are 0-1)
                decoder_layers.append(nn.Sigmoid())
            else:
                # Hidden layers: ReLU
                decoder_layers.append(nn.ReLU())
                
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def get_encoding(self, x):
        with torch.no_grad():
            return self.encoder(x)

    def decode(self, encoding):
        with torch.no_grad():
            # encoding might be numpy, convert to tensor
            if not isinstance(encoding, torch.Tensor):
                encoding = torch.tensor(encoding).float()
            # Check device of model
            device = next(self.parameters()).device
            encoding = encoding.to(device)
            
            return self.decoder(encoding)