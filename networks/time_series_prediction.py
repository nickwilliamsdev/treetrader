import torch
import torch.nn as nn

# --- Positional Encoding (Corrected) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Add batch dimension for broadcasting
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # Add positional encoding to the input embeddings
        # Select positional encodings up to the sequence length of x
        return x + self.pe[:, :x.size(1), :]


# --- 2. Model Architecture (Modified) ---
class StockPredictor(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dropout, max_seq_len):
        super(StockPredictor, self).__init__()
        self.d_model = d_model

        # Linear layer to project input features to the Transformer's d_model
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional Encoding layer
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len)

        # Define the Transformer Encoder layer with causal attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            # We will generate the causal mask in the forward pass
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # A simple linear head for the final prediction
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, src):
        # src shape: (batch_size, seq_len, input_dim)

        # Project input features to match d_model
        src = self.input_projection(src) # shape: (batch_size, seq_len, d_model)

        # Add positional encoding
        src = self.positional_encoding(src)

        # Generate a causal mask to prevent attention to future tokens.
        # This is CRITICAL for time-series prediction.
        seq_len = src.size(1)
        src_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(src.device)

        # Pass the input and the mask through the transformer encoder
        transformer_output = self.transformer_encoder(src, mask=src_mask)

        # Take the output of the last token in the sequence for prediction
        # The last token has access to the entire input sequence history
        last_token_output = transformer_output[:, -1, :]

        # Pass the last token output through the prediction head
        prediction = self.output_head(last_token_output)

        # Squeeze to remove the extra dimension of size 1
        return prediction.squeeze(-1)