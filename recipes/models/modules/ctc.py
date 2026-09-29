import torch

from .utils import MLP, LLaMaMLP

class CTC_Module(torch.nn.Module):
    def __init__(
        self,
        input_dim:int, # backbone dim
        vocab_size:int # vocabulary size
    ):
        super().__init__()

        self.project = torch.nn.Linear(input_dim, vocab_size)

    def forward(self, x):
        output = self.project(x)
        return output

class CTC_Gate_Module(torch.nn.Module):
    def __init__(
        self,
        hidden_dim:int,
        output_dim:int, # backbone dim
        vocab_size:int, # vocabulary size
    ):
        super().__init__()

        self.linear = torch.nn.Linear(vocab_size, hidden_dim)
        self.conv1d = torch.nn.Sequential(
            torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, groups=hidden_dim, padding=1),
            torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        )

        self.mlp_concated = MLP(
            input_dim=(output_dim + hidden_dim),
            output_dim=output_dim
        )
        self.linear_residual = torch.nn.Linear(output_dim, output_dim)
        self.mlp = LLaMaMLP(output_dim, output_dim*4)

        self.act = torch.nn.SiLU()

        self.project = torch.nn.Linear(output_dim, vocab_size)

    def forward(self, x, ctc_logits):
        # x: (B, T, Dim)
        # ctc_logits: (B, T, Vocab)
        ctc_logits = self.linear(ctc_logits) # (B, T, H)
        ctc_logits = self.conv1d(ctc_logits.transpose(1, 2)).transpose(1, 2) # (B, H, T)

        x_concated = torch.cat([x, ctc_logits], dim=-1) # (B, T, D+H)
        x_concated = self.mlp_concated(x_concated) # (B, T, D)

        x_residual = self.linear_residual(x) # (B, T, D)

        x = (self.act(x_concated) * x) + x_residual

        x = self.mlp(x) # (B, T, D)

        output = self.project(x) # (B, T, V)

        return x, output

class CTC(torch.nn.Module):
    def __init__(
        self,
        input_dim:int, # backbone dim
        hidden_dim:int,
        output_dim:int, # backbone dim
        vocab_size:int, # vocabulary size
    ):
        super().__init__()

        self.ctc = CTC_Module(
            input_dim=input_dim,
            vocab_size=vocab_size
        )

        self.ctc_gate = CTC_Gate_Module(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            vocab_size=vocab_size
        )
    
    def forward(self, x):
        # x: (B, T, Dim)
        ctc_logits = self.ctc(x) # (B, T, Vocab)

        audio_embeds, ctc_gate_logits = self.ctc_gate(x, ctc_logits) # (B, T, Dim), (B, T, Vocab)

        return audio_embeds, ctc_logits, ctc_gate_logits