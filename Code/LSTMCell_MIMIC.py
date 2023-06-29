# Needed Libraries

import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, input_size):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Linear(input_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        #print('shape of input:', inputs.shape)
        scores = self.linear(inputs).squeeze(-1)
        attention_weights = self.softmax(scores)
        weighted_output = inputs * attention_weights.unsqueeze(-1)
        output = torch.sum(weighted_output, dim=1)
        return output, attention_weights


# LSTMCell class
class LSTMCell(nn.Module):
    def __init__(self, hidden_size: int,
                number_features: int, n_class: int, device: None, sr: list, use_sr: bool = False,
                final_decay: bool = False, final_only_ct: bool = False, num_layers: int = 1, 
                bias: bool = True, if_relu_at_end: bool = False, if_dropout: bool = False, 
                dropout = 0.2, if_static = False, aggregate_by='mean', if_decay = True, final_no_ct: bool  = False):
        
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.number_features = number_features
        self.n_class = n_class
        self.bias = bias
        self.device = device
        self.use_sr = use_sr
        self.final_decay = final_decay
        self.final_only_ct = final_only_ct
        self.final_no_ct = final_no_ct
        self.if_relu_at_end = if_relu_at_end
        self.if_decay = if_decay
        self.aggregate_by = aggregate_by
        if use_sr:
            self.sr = torch.tensor(sr)
        else:
            self.sr = torch.ones(self.number_features)

        self.layers_per_features = nn.ModuleList([nn.Linear(self.hidden_size+1, 
            self.hidden_size*4, bias=self.bias) for i in range(self.number_features)]).to(self.device)
        self.decay_per_features = nn.ModuleList([nn.Linear(1, 1, 
            bias=self.bias) for i in range(self.number_features)]).to(self.device)
        if self.final_only_ct:
            in_for_out_dim = self.hidden_size
        elif self.final_no_ct:
            in_for_out_dim = self.number_features*self.hidden_size
        else:
            in_for_out_dim = self.number_features*self.hidden_size + self.hidden_size

        self.output_layers = nn.Linear(in_for_out_dim, self.n_class).to(self.device)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_for_out_dim, in_for_out_dim),
            nn.ReLU(),
            nn.Linear(in_for_out_dim, self.n_class),
        )

        self.if_dropout = if_dropout
        self.dropout = nn.Dropout(dropout)
        self.attention_layer = AttentionLayer(self.hidden_size)

        # if dataset != 'mimic':
        #     self.static_feat = True
        #     if dataset == 'p12':
        #         d_static = 8
        #     elif dataset == 'p19':
        #         d_static = 6
        #     self.emb = nn.Linear(d_static, self.hidden_size)
        # else:
        #     self.static_feat=False

        # if not self.static_feat:
        #     self.output_layers = nn.Linear(in_for_out_dim, self.n_class).to(self.device)
        # else:
        #     # additional encoding of static feature into a vector of self.hidden_size
        #     self.output_layers = nn.Sequential(
        #                             nn.Linear(self.number_features*self.hidden_size + self.hidden_size*2, self.hidden_size*4),
        #                             nn.ReLU(),
        #                             nn.Linear(self.hidden_size*4, self.n_class)
        #                             ).to(self.device)


    # Aggregate function on list of tensors
    def agg_func(self, list_tensor, fn = "mean"):
        if fn == "mean":
            return torch.mean(torch.stack(list_tensor), dim=0)
        elif fn == 'max':
            return torch.max(torch.stack(list_tensor), dim=0).values
        elif fn == 'attention':
            out, wts = self.attention_layer(torch.stack(list_tensor).view(1,-1, self.hidden_size))
            #print("attention applied shape:", out.shape)
            return out.squeeze(0)
    
    # Decay function
    def decay_func(self, feat_index, time_def, cur_device):
        if self.if_decay:
            tensor_zero = torch.tensor(0.).to(cur_device)
            decay_module_val = self.decay_per_features[feat_index](time_def*self.sr[feat_index])
            decay = torch.exp(-1*torch.max(tensor_zero, decay_module_val))
            h_t_decayed = decay*self.h_t[feat_index,:].clone()
        else:
            h_t_decayed = self.h_t[feat_index,:].clone()

        return h_t_decayed
        
    def forward(self, X, lengths, init_states = None, training = False):
        # X has four featutres - t, m, z, d
        # t - t denotes the time value and it is sorted
        prediction = []
        for i in range(X.shape[0]):

            t = X[i, 0, :lengths[i]]
            m = X[i, 1, :lengths[i]].long()
            x = X[i, 2, :lengths[i]]
            delt = X[i, 3, :lengths[i]].float()

            self.h_t = torch.zeros(self.number_features, self.hidden_size, dtype=torch.float).to(t.device)
            # Since c_t is common for every feature
            self.c_t = torch.zeros(self.hidden_size, dtype=torch.float).to(t.device)

            # Grouing the features received at each time instance
            # time_group = pd.Series(range(len(t))).groupby(t, sort=False).apply(list).tolist()
            _, counts = torch.unique_consecutive(t, return_counts = True)
            
            it = 0
            for t_i in counts: # time_group:
                agg_c_t = []
                for j in range(it, it + t_i): # t_i
                    it = it + 1
                    h_t_decayed = self.decay_func(m[j], delt[j:j+1], t.device)
                    if self.if_dropout and training:
                        h_t_decayed = self.dropout(h_t_decayed)
                    # tensor_zero = torch.tensor(0.).to(t.device)
                    # decay_module_val = self.decay_per_features[m[j]](delt[j:j+1]*self.sr[m[j]])
                    # decay = torch.exp(-1*torch.max(tensor_zero, decay_module_val))
                    # h_t_decayed = decay*self.h_t[m[j],:].clone()
                    inp_i = torch.cat((x[j:j+1], h_t_decayed)).float()
                    out_i = self.layers_per_features[m[j]](inp_i)
                    gate_input, gate_forget, gate_output, gate_pre_c, = out_i.chunk(4)
                    gate_input = torch.sigmoid(gate_input)
                    gate_forget = torch.sigmoid(gate_forget)
                    gate_output = torch.sigmoid(gate_output)
                    gate_pre_c = torch.tanh(gate_pre_c)
                    agg_c_t.append(gate_forget * self.c_t + gate_input * gate_pre_c)
                    self.h_t[m[j],:] = gate_output*torch.tanh(agg_c_t[-1])

                if len(agg_c_t) == 1:
                    self.c_t = agg_c_t[0]
                else:
                    self.c_t = self.agg_func(agg_c_t, self.aggregate_by)

            # For output, a softmax class is applied between the concatenation of the LSTM cell for each feature
            if self.final_decay:
                for feat_i in range(self.number_features):
                    if (m == feat_i).nonzero().numel() != 0:
                        delt_final = (torch.max(t) - t[(m == feat_i).nonzero()[-1].item()]).reshape(1).float()
                        # decay = torch.exp(-1*torch.max(torch.tensor(0.).to(t.device),
                        #         self.decay_per_features[feat_i](delt_final*self.sr[m[feat_i]])))
                        # h_t_decayed = decay*self.h_t[feat_i,:].clone()
                        h_t_decayed = self.decay_func(feat_i, delt_final, t.device)
                        self.h_t[feat_i,:] = h_t_decayed
                # Decay final h_t for output prediction
 
            if self.final_only_ct:
                input_for_out = torch.reshape(self.c_t.reshape([1,self.hidden_size]), [-1])
            elif self.final_no_ct:
                input_for_out = torch.reshape(self.h_t, [-1])
            else:
                input_for_out = torch.reshape(torch.cat((self.c_t.reshape([1,self.hidden_size]), self.h_t)), [-1])

            if self.if_relu_at_end:
                pred = torch.softmax(self.mlp(input_for_out), dim = 0)
            else:
                pred = torch.softmax(self.output_layers(input_for_out), dim = 0)
            prediction.append(pred)
            
        return torch.stack(prediction)