import torch
import torch.nn as nn
from constants import d_a,d_b
import torch.nn.functional as F
class AttentionLayer(nn.Module):
    
    def __init__(self,size,n_labels,attention_mode):
        super(AttentionLayer, self).__init__()
        self.attention_mode = attention_mode
        self.size = size
        self.d_a = d_a
        self.d_b = d_b
        self.n_labels = n_labels
        self.r = n_labels
        self.first_linear = nn.Linear(self.size, self.d_a, bias=False)
        self.second_linear = nn.Linear(self.d_a, n_labels, bias=False)
        self.third_linear = nn.Linear(self.size, self.d_b, bias=False)
        self._init_weights(mean=0.0, std=0.03)

    def _init_weights(self, mean=0.0, std=0.03) -> None:
        torch.nn.init.normal_(self.first_linear.weight, mean, std)
        torch.nn.init.normal_(self.second_linear.weight, mean, std)
        torch.nn.init.normal_(self.third_linear.weight, mean, std)

    def forward(self, x,label_weights):
        weights = torch.tanh(self.first_linear(x))
        att_weights = self.second_linear(weights)
        att_weights = F.softmax(att_weights, 1).transpose(1, 2)
        if len(att_weights.size()) != len(x.size()):
            att_weights = att_weights.squeeze()
        weighted_output =  att_weights @ x
        weighted_output = torch.tanh(self.third_linear(weighted_output))
        num_channels  = label_weights.shape[2]
        weighted_output = weighted_output.unsqueeze(2).repeat(1,1,num_channels,1)
        new_output = torch.mul(label_weights,weighted_output)
        new_output = torch.mul(torch.sum(new_output,axis=2),1.0/num_channels)                     
        
        return new_output