import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearVAE(nn.Module):
    def __init__(self, features, input_size, extra_decoder_input, reconstruct_size, nhead, num_layers, dim_feedforward):
        super(LinearVAE, self).__init__()
        self.features = features

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=reconstruct_size, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Encoder fully connected layers
        self.encoder_fc = nn.Linear(in_features=input_size, out_features=2*features)

        # Decoder fully connected layers
        self.decoder_fc = nn.Linear(in_features=features + extra_decoder_input, out_features=reconstruct_size)

    def reparameterize(self, mu, log_var):
        # ... (same as before)
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def encode(self, src):
        # src shape is (seq_len, batch, input_size)
        memory = self.transformer_encoder(src)
        # We take the last output for the VAE
        last_memory = memory[-1]  # Shape: (batch, input_size)
        encoded = self.encoder_fc(last_memory)
        mu = encoded[:, :self.features]
        log_var = encoded[:, self.features:]
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def forward(self, src, tgt, memory):
        # src shape is (seq_len, batch, input_size)
        # tgt shape is (tgt_seq_len, batch, input_size)
        # memory is the output of the encoder (seq_len, batch, input_size)
        z, mu, log_var = self.encode(src)

        # Prepare the latent vector z to be concatenated with tgt for the transformer decoder
        # This often involves repeating z to match the sequence length of tgt
        z = z.unsqueeze(0).repeat(tgt.size(0), 1, 1)
        dec_input = torch.cat([z, tgt], dim=2)
        
        # Pass dec_input through the transformer decoder along with memory
        output = self.transformer_decoder(tgt=dec_input, memory=memory)
        reconstruction = self.decoder_fc(output)
        return reconstruction, mu, log_var
