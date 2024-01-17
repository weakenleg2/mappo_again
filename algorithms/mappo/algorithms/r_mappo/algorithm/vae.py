import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearVAE(nn.Module):
    def __init__(self, features, input_size, extra_decoder_input, reconstruct_size):
        super(LinearVAE, self).__init__()
        HIDDEN=64
        self.features = features
        self.num_layers = 2
        # encoder
        # self.gru = nn.GRU(input_size=input_size, hidden_size=HIDDEN, batch_first=True) # not used for now
        
        # self.encoder = nn.Sequential(
        #     nn.Linear(in_features=input_size, out_features=HIDDEN),
        #     nn.ReLU(),
        #     nn.Linear(in_features=HIDDEN, out_features=2*features)
        # )
        self.gru = nn.GRU(input_size=input_size, hidden_size=HIDDEN, batch_first=True)
        
        # Encoder fully connected layers
        self.encoder = nn.Sequential(
            nn.Linear(in_features=HIDDEN, out_features=2*features)
        )
        # var and mean
        # self.decoder = nn.Sequential(
        #     nn.Linear(in_features=features + extra_decoder_input, out_features=HIDDEN),
        #     nn.ReLU(),
        #     nn.Linear(in_features=HIDDEN, out_features=HIDDEN),
        #     nn.ReLU(),
        #     nn.Linear(in_features=HIDDEN, out_features=reconstruct_size),
        # )
        self.decoder_rnn = nn.GRU(input_size=features + extra_decoder_input, hidden_size=HIDDEN, num_layers=self.num_layers, batch_first=True)
        self.decoder = nn.Linear(HIDDEN, reconstruct_size)
 
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def encode(self, x):
        x, _ = self.gru(x)
        x = self.encoder(x)
        mu = x[:, :self.features]
        log_var = x[:, self.features:]
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        return mu
        
    def forward(self, x, xp):
        # encoding
        x, _ = self.gru(x)
        # encode input obs transition and reward function
        x = self.encoder(x)
        
        mu = x[: , :self.features]
        log_var = x[:, self.features:]
        
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        # figure 2, z
        # decoder_hidden = z.unsqueeze(0).repeat(self.num_layers, 1, 1)

        dec_input = torch.cat([z, xp], axis=-1)
        # print("z",z.shape)
        # print("xp",xp.shape)
        reconstructed, _ = self.decoder_rnn(dec_input)
  
        # xp, obs and action    
        # decoding
        # reconstruction, latent part from encoder,obs and reward
        reconstruction = self.decoder(reconstructed)
        return reconstruction, mu, log_var