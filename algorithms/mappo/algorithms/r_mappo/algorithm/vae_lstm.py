import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




class LinearVAE(nn.Module):
    def __init__(self, features, input_size, extra_decoder_input, reconstruct_size):
        super(LinearVAE, self).__init__()
        HIDDEN = 64
        self.features = features
        self.num_layers = 2
        # encoder
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=HIDDEN, batch_first=True)

        # Encoder fully connected layers
        self.encoder = nn.Sequential(
            nn.Linear(in_features=HIDDEN, out_features=2*features)
        )

        # LSTM decoder
        self.decoder_lstm = nn.LSTM(input_size=features + extra_decoder_input, 
                                  hidden_size=HIDDEN, num_layers=self.num_layers) 
        self.decoder = nn.Sequential(nn.Linear(HIDDEN, reconstruct_size))

    def reparameterize(self, mu, log_var):
        # ... (same as before)
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def encode(self, x):
        x, _ = self.lstm(x)  # LSTM returns hidden state and cell state
        # Only the hidden state is passed through the encoder
        # only select out
        x = self.encoder(x)  # Use the last layer's hidden state
        mu = x[:, :self.features]
        log_var = x[:, self.features:]
        z = self.reparameterize(mu, log_var)
        return mu

    def forward(self, x, xp):
        # encoding
        x, _ = self.lstm(x)  # LSTM returns hidden state and cell state
        x = self.encoder(x)  # Use the last layer's hidden state

        mu = x[:, :self.features]
        log_var = x[:, self.features:]
        z = self.reparameterize(mu, log_var)

        dec_input = torch.cat([z, xp], axis=-1)
        reconstructed, _ = self.decoder_lstm(dec_input)
        reconstruction = self.decoder(reconstructed)
        return reconstruction, mu, log_var
