import torch
import torch.nn as nn
import numpy as np

class MDense(nn.Module):
    def __init__(self, input_features, output_features, use_bias=True):
        super(MDense, self).__init__()

        self.input_features = input_features
        self.output_features = output_features
        self.use_bias=use_bias

        self.weight1 = nn.Parameter(torch.Tensor(output_features, input_features).zero_())
        nn.init.xavier_uniform_(self.weight1 )
        self.weight2 = nn.Parameter(torch.Tensor(output_features, input_features).zero_())
        nn.init.xavier_uniform_(self.weight2)
        if use_bias:
            self.bias1 = nn.Parameter(torch.Tensor(output_features).zero_())
            self.bias2 = nn.Parameter(torch.Tensor(output_features).zero_())
        else:
            self.register_parameter('bias', None)
        self.factor1 = nn.Parameter(torch.ones(output_features))
        self.factor2 = nn.Parameter(torch.ones(output_features))

    def forward(self, inputs):
        output1 = inputs.matmul(self.weight1.t())
        output2 = inputs.matmul(self.weight2.t())
        if self.use_bias:
            output1 = output1 + self.bias1
            output2 = output2 + self.bias2
        output1 = torch.tanh(output1) * self.factor1
        output2 = torch.tanh(output2) * self.factor2
        output = output1 + output2
        return output
