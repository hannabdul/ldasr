import torch
from torch import nn
from torch import Tensor
from models.model.conformer import Conformer

from models.embedding.positional_encoding import PositionalEncoding

torch.set_printoptions(profile='full')

class Conv1dSubampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv1dSubampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=0, padding_mode='zeros'),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=0, padding_mode='zeros')
        )
    def forward(self, inputs: Tensor) -> torch.tensor:
        outputs = self.sequential(inputs)
        return outputs

class Conformer_with_LayerDrop(nn.Module):
    ### This model contains single linear layer as a decoder at the end of last encoder
    def __init__(self, src_pad_idx, n_enc_replay, 
                 enc_voc_size, dec_voc_size, d_model, 
                 n_head, max_len,  dim_feed_forward, 
                 n_encoder_layers,  features_length, 
                 drop_prob, depthwise_kernel_size, device, 
                 gate_values = torch.ones(12,), 
                 flag_use_single_out = False, 
                 flag_use_gating = False):
        """ Flag_use_single_out: If False, uses linear decoders as specified at multiple Early-Exits, 
                                if True, uses Only ONE Linear Decoder at the end of model
            Flag_use_gating: if False, no Gating is applied to the model,
                            If True, Gating Mechanism is applied to the model. """
        super().__init__()
        self.input_dim = d_model
        self.num_heads = n_head
        self.ffn_dim = dim_feed_forward
        self.num_layers = n_encoder_layers
        self.depthwise_conv_kernel_size = depthwise_kernel_size
        self.n_enc_replay = n_enc_replay
        self.dropout = drop_prob
        self.device = device
        self.gate_values = gate_values
        # self.gate = torch.bernoulli(torch.randint(1, (1,)), p = gate_prob).item()
        self.flag_use_single_out = flag_use_single_out
        self.flag_use_gating = flag_use_gating
        
        self.conv_subsample = Conv1dSubampling(in_channels = features_length,
                                               out_channels = d_model)
        self.positional_encoder = PositionalEncoding(d_model = d_model,
                                                     dropout = drop_prob,
                                                     max_len = max_len)
        self.single_linear = nn.Linear(d_model, dec_voc_size)
        self.linears = nn.ModuleList([nn.Linear(d_model, dec_voc_size) for _ in range(self.n_enc_replay)])
        # self.conformer = nn.ModuleList([Conformer(input_dim=self.input_dim, num_heads=self.num_heads, ffn_dim=self.ffn_dim, num_layers=self.num_layers, depthwise_conv_kernel_size=self.depthwise_conv_kernel_size, dropout=self.dropout) for _ in range(self.n_enc_replay)])
        self.conformer = nn.ModuleList([Conformer(input_dim = self.input_dim, 
                                                  num_heads = self.num_heads,
                                                  ffn_dim = self.ffn_dim,
                                                  num_layers = self.num_layers,
                                                  depthwise_conv_kernel_size = self.depthwise_conv_kernel_size,
                                                  dropout = self.dropout,
                                                  # gate_prob = self.gate_prob,
                                                  flag_use_gating = self.flag_use_gating) for _ in range(self.n_enc_replay)])
        
    """### Adding code for testing different combinations of gating
    def set_gate_values(self, gate_values, flag_print = False):
        for i in range(len(self.conformer)):
            self.conformer[i].set_gate_value1(gate_values[i].item(), flag_print)
        if flag_print:
            print("In Model Class - New Value of Gate is : ,", gate_values)"""
    
    def train_generate_gate_values(self, brv_prob):
        gate_values = torch.bernoulli(torch.randint(1, (12,)), p = brv_prob)
        self.set_gate_values(gate_values)
    
    def set_gate_values(self, gate_values):
        for i in range(len(self.conformer)):
            for j in range(2):
                self.conformer[i].conformer_layers[j].gate = gate_values[i * 2 + j].item()
                
    def print_gate_values(self):
        g_val = []
        for i in range(len(self.conformer)):
            for j in range(2):
                g_val.append(self.conformer[i].conformer_layers[j].gate)
        print("Current gate values are ", g_val)
        # return g_val
    
    def forward(self, src, lengths):
        src = self.conv_subsample(src)      # Convolutional Feature Extractor
        src = self.positional_encoder(src.permute(0,2,1))   # Positional Encoding

        length = torch.clamp(lengths/4, max = src.size(1)).to(torch.int).to(self.device)
        enc_out = []
        enc = src
        if self.flag_use_single_out:
            for layer in self.conformer:
                enc, _ = layer(enc, length)
                # print("Encoder's output shape is: ", enc.size())                
            out = self.single_linear(enc)
            # print("After Linear shape is : ", out.size())
            out = torch.nn.functional.log_softmax(out, dim = 2)
            # print("After Log-Softmax shape is : ", out.size())
            enc_out += [out.unsqueeze(0)]       # Out Shape : torch.Tensor([1, 12, 390, 256]) (number_of_exits, batch_size, number of audio chunks, d_model)
            return enc_out[0]
        else:
            for linear, layer  in zip(self.linears, self.conformer):
                # enc_residual = enc
                enc, _ = layer(enc, length)
                # print("Encoder's output shape is: ", enc.size())
                out = linear(enc)
                # print("After Linear shape is : ", out.size())
                out = torch.nn.functional.log_softmax(out, dim = 2)
                # print("After Log-Softmax shape is : ", out.size())
                enc_out += [out.unsqueeze(0)]
            enc_out = torch.cat(enc_out)            # Out Shape : torch.Tensor([6, 12, 390, 256]) (number_of_exits, batch_size, x, d_model)
            # print("After concatenating shape is: ", enc_out.size())
            return enc_out