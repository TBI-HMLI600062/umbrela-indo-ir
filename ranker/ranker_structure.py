import torch
import torch.nn as nn

# Consider adding two layers on context to allow modification of base model representations that are not suitable


class RankModel_Transformer(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_attention_heads,
        dropout_prob,
        num_layers,
        # num_pre,
        half,
    ):
        super(RankModel_Transformer, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)

        # self.ffn = MLP(hidden_size, num_layers=num_pre)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_attention_heads, dropout_prob)
                for _ in range(num_layers)
            ]
        )

        self.output_weights = nn.Parameter(torch.empty(hidden_size, 1))  # hidden_size))

        # Use Xavier initialization
        nn.init.xavier_uniform_(self.output_weights)

        # Initialize all linear layers
        self._init_weights()

        # Convert to float16 if half is True
        if half:
            self.to(torch.bfloat16)

    def _init_weights(self):
        for layer in self.layers:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, candidate_hstates, context_hstates, key_padding_mask=None):
        # Ensure context_hstates has a batch dimension, and permute candidate_hstates

        candidate_hstates = self.linear(candidate_hstates)
        context_hstates = self.linear(context_hstates)

        context_hstates = context_hstates.unsqueeze(
            0
        )  # Add batch dim (1, batch_size, hidden_size)
        candidate_hstates = candidate_hstates.permute(
            1, 0, 2
        )  # (seq_len, batch_size, input_size)

        x = torch.cat((context_hstates, candidate_hstates), dim=0)
        # if candidate_mask is not None:
        #     context_mask = torch.zeros((candidate_mask.shape[0], 1), dtype=torch.bool)
        #     key_padding_mask = torch.cat((context_mask, candidate_mask), dim=1)
        # else:
        #     key_padding_mask = None
        for layer in self.layers:
            x = layer(x, key_padding_mask)

        # After attention, output x will have shape (seq_len, batch_size, hidden_size)
        logits = torch.matmul(x, self.output_weights)
        # logits = logits.squeeze(-1)
        logits = logits.squeeze(-1).permute(1, 0)
        logits = logits[:, 1:]
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super(TransformerBlock, self).__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_attention_heads, dropout=dropout_prob
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout_prob),
        )
        self.norm3 = nn.LayerNorm(hidden_size)

    def forward(self, x, key_padding_mask=None):
        # Cross Attention: Ensure context_hstates has correct shape for attention (seq_len, batch_size, hidden_size)
        attn_output, _ = self.self_attention(x, x, x, key_padding_mask=key_padding_mask)

        # Add residual connection and normalization
        x = self.norm2(x + attn_output)

        # Feed Forward Network
        output = x + self.ffn(x)
        # output = self.ffn(x)
        output = self.norm3(output + x)

        return output


class MLP(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(MLP, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class RankModel_MLP(nn.Module):
    def __init__(
        self,
        hidden_size,
        output_size,
        num_layers,
        dropout_prob=0.1,
        criterion=None,
        half=True,
    ):
        super(RankModel_MLP, self).__init__()
        self.criterion = criterion
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        self.layers = nn.ModuleList([self.ffn for _ in range(num_layers)])
        self.projection = nn.Linear(hidden_size, output_size)

        self._init_weights()

        if half:
            self.to(torch.bfloat16)

    def _init_weights(self):
        for layer in self.layers:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        x = self.projection(x)
        if self.criterion is None:
            return x[:, 1:, 0]
        elif "cos" in self.criterion:
            t = float(self.criterion.split("_t")[-1])
            x = nn.functional.normalize(x, p=2.0, dim=-1, eps=1e-4)
            x = torch.einsum("bd,bld->bl", x[:, 0], x[:, 1:])
            return x / t
        else:
            raise NotImplementedError
