import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth, init_std=1e-6, 
                 output_dim = None):
        assert(depth >= 0)
        super(ResNet, self).__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        if output_dim is None:
            output_dim = hidden_dim

        self.linear_out = nn.Linear(hidden_dim, output_dim)
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim, init_std) for i in range(depth)])

    def forward(self, input):
        input = input.view(input.size(0), -1)
        x = self.linear_in(input)
        for res_block in self.res_blocks:
            x = res_block(x)
        return self.linear_out(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim, noise=1e-6):
        super(ResidualBlock, self).__init__()
        self.noise = noise
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim, bias = False)
        self.l1.bias.data.uniform_(-self.noise,self.noise)
        self.l1.weight.data.uniform_(-self.noise,self.noise) 
        self.l2.weight.data.uniform_(-self.noise,self.noise)

    def forward(self, x):
        return x + self.l2(F.relu(self.l1(x)))

    
class SumLayers(nn.Module):
    def __init__(self, input_D, input_max, filter_n, layer_count):
        super(SumLayers, self).__init__()
        self.fc1 = nn.Linear(input_D, filter_n)
        self.relu1 = nn.ReLU()
        self.fc_blocks = nn.ModuleList([nn.Sequential(nn.Linear(filter_n, filter_n), nn.ReLU()) for _ in range(layer_count-1)])
        
    def forward(self, X, present):
        xt = X  
        x = self.fc1(xt) 
        x = self.relu1(x)
        for fcre in self.fc_blocks:
            x = fcre(x)
        
        x = (present.unsqueeze(-1) * x)
        return x.sum(1)

    

class ClusterCountingNetwork(nn.Module):
    def __init__(self, input_D, input_max, 
                 sum_fc_filternum, sum_fc_layercount, 
                 post_sum_fc, post_sum_layercount, 
                 output_dim):
        super(ClusterCountingNetwork, self).__init__()

        self.sum_layer = SumLayers(input_D, input_max, sum_fc_filternum, 
                                  sum_fc_layercount)
        self.post_sum = nn.Sequential(ResNet(sum_fc_filternum, 
                                             post_sum_fc,
                                             post_sum_layercount, 
                                         ),
                                   nn.Linear(post_sum_fc, output_dim), 
                                   nn.ReLU()
                                     )
    def forward(self, X, present):
        sum_vals = self.sum_layer(X, present)
        return self.post_sum(sum_vals)


class ResNetRegression(nn.Module):
    def __init__(self, D, block_sizes, INT_D, FINAL_D, 
                 use_batch_norm=False, OUT_DIM=1):
        super(ResNetRegression, self).__init__()

        layers = [nn.Linear(D, INT_D)]

        for block_size in block_sizes:
            layers.append(ResNet(INT_D, INT_D, block_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(INT_D))
        layers.append(nn.Linear(INT_D, FINAL_D))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(FINAL_D, OUT_DIM))
                
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X) 


class PyTorchResNet(nn.Module):
    def __init__(self, block, layers, input_img_size = 64, 
                 num_channels=3, 
                 num_classes=1, first_kern_size=7, 
                 final_avg_pool_size=7, inplanes=64):
        self.inplanes = inplanes
        super(PyTorchResNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, self.inplanes, kernel_size=first_kern_size, 
                               stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block_layers = []
        for i, l in enumerate(layers):
            stride = 1 if i == 0 else 2 
            layer = self._make_layer(block, 64*2**i, l, stride=stride)
            self.block_layers.append(layer)

        self.block_layers_seq = nn.Sequential(*self.block_layers)

        last_image_size = input_img_size // (2**(len(layers) +1))
        post_pool_size = last_image_size - final_avg_pool_size+1
        self.avgpool = nn.AvgPool2d(final_avg_pool_size, stride=1, padding=0)
        expected_final_planes = 32 * 2**len(layers)

        self.fc = nn.Linear(expected_final_planes * block.expansion * post_pool_size**2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.block_layers_seq(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class SimpleGraphModel(nn.Module):
    def __init__(self, MAX_N, input_feature_n, 
                 output_features_n, noise=1e-5, 
                 single_out_row=True, batch_norm = False, input_batch_norm=False):
        super(SimpleGraphModel, self).__init__()
        self.MAX_N = MAX_N
        self.input_feature_n = input_feature_n
        self.output_features_n = output_features_n
        self.noise = noise
        self.linear_layers = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.use_batch_norm = batch_norm
        self.input_batch_norm = input_batch_norm
        if self.input_batch_norm:
            self.input_batch_norm_layer = nn.BatchNorm1d(input_feature_n)

        for i in range(len(output_features_n)):
            if i == 0:
                lin = nn.Linear(input_feature_n, output_features_n[i])
            else:
                lin = nn.Linear(output_features_n[i-1], output_features_n[i])
            lin.bias.data.uniform_(-self.noise,self.noise)
            lin.weight.data.uniform_(-self.noise,self.noise) 

            self.linear_layers.append(lin)
            self.relus.append(nn.ReLU())
            if self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(output_features_n[i] ))

        self.single_out_row = single_out_row

    def forward(self, args):
        (G, x, tgt_out_rows) = args
        if self.input_batch_norm:
            x = self.input_batch_norm_layer(x.reshape(-1, self.input_feature_n))
            x = x.reshape(-1, self.MAX_N, self.input_feature_n)
            
        for l in range(len(self.linear_layers)):
            x = self.linear_layers[l](x)
            x = torch.bmm(G, x)
            x = self.relus[l](x)
            if self.use_batch_norm:
                x = self.batch_norms[l](x.reshape(-1, self.output_features_n[l]))
                x = x.reshape(-1, self.MAX_N, self.output_features_n[l])
        if self.single_out_row:
            return torch.stack([x[i, j] for i, j in enumerate(tgt_out_rows)])
        else:
            return x
        

class ResGraphModel(nn.Module):
    def __init__(self, MAX_N, input_feature_n, 
                 output_features_n, noise=1e-5, 
                 single_out_row=True, batch_norm = False, 
                 input_batch_norm=False, resnet=True):
        super(ResGraphModel, self).__init__()
        self.MAX_N = MAX_N
        self.input_feature_n = input_feature_n
        self.output_features_n = output_features_n
        self.noise = noise
        self.linear_layers = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.use_batch_norm = batch_norm
        self.input_batch_norm = input_batch_norm
        self.use_resnet = resnet

        if self.input_batch_norm:
            self.input_batch_norm_layer = nn.BatchNorm1d(input_feature_n)

        for i in range(len(output_features_n)):
            if i == 0:
                lin = nn.Linear(input_feature_n, output_features_n[i])
            else:
                lin = nn.Linear(output_features_n[i-1], output_features_n[i])
            lin.bias.data.uniform_(-self.noise,self.noise)
            lin.weight.data.uniform_(-self.noise,self.noise) 

            self.linear_layers.append(lin)
            self.relus.append(nn.ReLU())
            if self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(output_features_n[i] ))

        self.single_out_row = single_out_row

    def forward(self, args):
        (G, x, tgt_out_rows) = args
        if self.input_batch_norm:
            x = self.input_batch_norm_layer(x.reshape(-1, self.input_feature_n))
            x = x.reshape(-1, self.MAX_N, self.input_feature_n)
            
        for l in range(len(self.linear_layers)):
            x1 = torch.bmm(G, self.linear_layers[l](x))
            x2 = self.relus[l](x1)
            
            if x.shape == x2.shape and self.use_resnet:
                x3 = x2 + x
            else:
                x3 = x2
            if self.use_batch_norm:
                x = self.batch_norms[l](x3.reshape(-1, self.output_features_n[l]))
                x = x.reshape(-1, self.MAX_N, self.output_features_n[l])
            else:
                x = x3
        if self.single_out_row:
            return torch.stack([x[i, j] for i, j in enumerate(tgt_out_rows)])
        else:
            return x
        
def goodmax(x, dim):
    return torch.max(x, dim=dim)[0]

class GraphMatLayer(nn.Module):
    def __init__(self, C, P , GS=1,  
                 noise=1e-6, agg_func=None, 
                 dropout=0.0):
        super(GraphMatLayer, self).__init__()

        self.GS = GS
        self.noise=noise

        self.linlayers = nn.ModuleList()
        self.dropout = dropout
        self.dropout_layers = nn.ModuleList()
        for ll in range(GS):
            l = nn.Linear(C, P)
            l.bias.data.normal_(0.0, self.noise)
            l.weight.data.normal_(0.0, self.noise) 
            self.linlayers.append(l)
            if dropout > 0.0:
                self.dropout_layers.append(nn.Dropout(p=dropout))
            
        self.r = nn.ReLU()
        self.agg_func = agg_func
 
    def forward(self, G, x):
        def apply_ll(i, x):
            y = self.linlayers[i](x)
            if self.dropout > 0:
                y = self.dropout_layers[i](y)
            return y

        multi_x = torch.stack([apply_ll(i,x) for i in range(self.GS)])
        xout = torch.stack([torch.matmul(G[i], multi_x[:, i]) for i in range(x.shape[0])])

        x = self.r(xout)
        if self.agg_func is not None:
            x = self.agg_func(x, dim=1)
        return x
        
class GraphMatLayers(nn.Module):
    def __init__(self, input_feature_n, 
                 output_features_n, resnet=False, GS=1, 
                 noise=1e-5, agg_func=None, dropout=0.0):
        super(GraphMatLayers, self).__init__()
        
        self.gl = nn.ModuleList()
        self.resnet = resnet

        for li in range(len(output_features_n)):
            if li == 0:
                gl = GraphMatLayer(input_feature_n, output_features_n[0],
                                   noise=noise, agg_func=agg_func, GS=GS, 
                                   dropout=dropout)
            else:
                gl = GraphMatLayer(output_features_n[li-1], 
                                   output_features_n[li], 
                                   noise=noise, agg_func=agg_func, GS=GS, 
                                   dropout=dropout)
            
            self.gl.append(gl)
    def forward(self, G, x):
        for gl in self.gl:
            x2 = gl(G, x)
            if self.resnet:
                if x.shape == x2.shape:
                    x3 = x2 + x
                else:
                    x3 = x2
            else:
                x3 = x2
            x = x3
        return x

class GraphMatHighwayLayers(nn.Module):
    def __init__(self, input_feature_n, 
                 output_features_n, resnet=False, GS=1, 
                 noise=1e-5, agg_func=None):
        super(GraphMatHighwayLayers, self).__init__()
        
        self.gl = nn.ModuleList()
        self.resnet = resnet

        for li in range(len(output_features_n)):
            if li == 0:
                gl = GraphMatLayer(input_feature_n, output_features_n[0],
                                   noise=noise, agg_func=agg_func, GS=GS)
            else:
                gl = GraphMatLayer(output_features_n[li-1], 
                                   output_features_n[li], 
                                   noise=noise, agg_func=agg_func, GS=GS)
            
            self.gl.append(gl)

    def forward(self, G, x):
        highway_out = []
        for gl in self.gl:
            x2 = gl(G, x)
            if self.resnet:
                if x.shape == x2.shape:
                    x3 = x2 + x
                else:
                    x3 = x2
            else:
                x3 = x2
            x = x3
            highway_out.append(x2)

        return x, torch.stack(highway_out, -1)


def batch_diagonal_extract(x):
    BATCH_N, M, _, N = x.shape
    return torch.stack([x[:, i, i, :] for i in range(M)], dim=1)

class GraphMatModel(nn.Module):
    def __init__(self, g_feature_n, g_feature_out_n, resnet=True, 
                 noise=1e-5, GS=1, OUT_DIM=1 ):
        super(GraphMatModel, self).__init__()
        self.gml = GraphMatLayers(g_feature_n, g_feature_out_n, 
                                  resnet=resnet, noise=noise, GS=GS)
        self.lin_out = nn.Linear(g_feature_out_n[-1], OUT_DIM )

    def forward(self, args):
        (G, x_G) = args
        G_features = self.gml(G, x_G)
        g_diag = batch_diagonal_extract(G_features)
        x_1 = self.lin_out(g_diag)
        return x_1


class GraphVertModel(nn.Module):
    def __init__(self, g_feature_n, g_feature_out_n=None, 
                 int_d = None, layer_n = None, 
                 resnet=True, 
                 init_noise=1e-5, agg_func=None, GS=1, OUT_DIM=1, 
                 batch_norm=False, out_std= False, 
                 resnet_out = False, resnet_blocks = (3, ), 
                 resnet_d = 128, 
                 graph_dropout=0.0, 
                 force_lin_init=False,
                 var_mode='relu_std',
                 init_var=1.0,
                 min_var=1e-4,
                 max_var=1000.0):
        
        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n

        super(GraphVertModel, self).__init__()
        self.gml = GraphMatLayers(g_feature_n, g_feature_out_n, 
                                  resnet=resnet, noise=init_noise, agg_func=parse_agg_func(agg_func), 
                                  GS=GS, dropout=graph_dropout)

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(g_feature_n)
        else:
            self.batch_norm = None

        self.resnet_out = resnet_out
        if not resnet_out:
            self.lin_out = nn.Linear(g_feature_out_n[-1], OUT_DIM)
        else:
            self.lin_out = ResNetRegression(g_feature_out_n[-1], 
                                                block_sizes = resnet_blocks, 
                                                INT_D = resnet_d, 
                                                FINAL_D=resnet_d, 
                                                OUT_DIM=OUT_DIM)

        self.out_std = out_std
        self.var_mode = var_mode
        self.min_var = float(min_var)
        self.max_var = float(max_var)
        self.init_var = float(init_var)

        if self.var_mode == 'softplus_var':
            if not (self.min_var < self.init_var <= self.max_var):
                raise ValueError(
                    f"Expected min_var < init_var <= max_var, got "
                    f"min_var={self.min_var}, init_var={self.init_var}, max_var={self.max_var}"
                )
            # Exact inverse of softplus(x) = init_var - min_var (following martius-lab/beta-nll),
            # using expm1 and an asymptotic linear branch to avoid float overflow for large delta.
            delta_var = self.init_var - self.min_var
            self.init_var_offset = (
                float(delta_var) if delta_var > 20.0 else float(np.log(np.expm1(delta_var)))
            )
        elif self.var_mode != 'relu_std':
            raise ValueError(f"Unknown var_mode: {self.var_mode}")

        if out_std:
            self.lin_out_std1 = nn.Linear(g_feature_out_n[-1], 128)
            self.lin_out_std2 = nn.Linear(128, OUT_DIM)

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, init_noise)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, args, return_g_features = False):
        (G, x_G) = args
        
        BATCH_N, MAX_N, F_N = x_G.shape

        if self.batch_norm is not None:
            x_G_flat = x_G.reshape(BATCH_N*MAX_N, F_N)
            x_G_out_flat = self.batch_norm(x_G_flat)
            x_G = x_G_out_flat.reshape(BATCH_N, MAX_N, F_N)
        
        G_features = self.gml(G, x_G)
        if return_g_features:
            return G_features

        g_squeeze = G_features.squeeze(1)
        g_squeeze_flat = g_squeeze.reshape(-1, G_features.shape[-1])
        
        if self.resnet_out:
            x_1 = self.lin_out(g_squeeze_flat).reshape(BATCH_N, MAX_N, -1)
        else:
            x_1 = self.lin_out(g_squeeze)
        if self.out_std:
            x_std = F.relu(self.lin_out_std1(g_squeeze))
            if getattr(self, 'var_mode', 'relu_std') == 'softplus_var':
                # Compute variance in FP32 for numerical stability under AMP autocast
                raw_var = self.lin_out_std2(x_std).float()
                var = F.softplus(raw_var + self.init_var_offset) + self.min_var
                var = torch.clamp(var, min=self.min_var, max=self.max_var)
                return {'mu': x_1, 'var': var, 'std': torch.sqrt(var)}
            else:
                x_1_std = F.relu(self.lin_out_std2(x_std))
                return {'mu' : x_1, 'std' : x_1_std}
        else:
            return x_1


class GraphVertResOutModel(nn.Module):
    def __init__(self, g_feature_n, g_feature_out_n, resnet=True, 
                 noise=1e-5, agg_func=None, GS=1, OUT_DIM=1, 
                 batch_norm=False, out_std= False, 
                 force_lin_init=False):
        
        super(GraphVertResOutModel, self).__init__()
        self.gml = GraphMatLayers(g_feature_n, g_feature_out_n, 
                                  resnet=resnet, noise=noise, agg_func=agg_func, 
                                  GS=GS)

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(g_feature_n)
        else:
            self.batch_norm = None

        print("g_feature_out_n[-1]=", g_feature_out_n[-1])

        self.lin_out = ResNetRegression(g_feature_out_n[-1], 
                                        block_sizes = [3], 
                                        INT_D = 128, 
                                        FINAL_D=1024, 
                                        OUT_DIM=OUT_DIM)

        self.out_std = out_std

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, noise)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, args):
        (G, x_G) = args
        
        BATCH_N, MAX_N, F_N = x_G.shape

        if self.batch_norm is not None:
            x_G_flat = x_G.reshape(BATCH_N*MAX_N, F_N)
            x_G_out_flat = self.batch_norm(x_G_flat)
            x_G = x_G_out_flat.reshape(BATCH_N, MAX_N, F_N)
        
        G_features = self.gml(G, x_G)
        g_squeeze = G_features.squeeze(1).reshape(-1, G_features.shape[-1])
        x_1 = self.lin_out(g_squeeze)
        return x_1.reshape(BATCH_N, MAX_N, -1)


def parse_agg_func(agg_func):
    if isinstance(agg_func, str):
        if agg_func == 'goodmax':
            return goodmax
        elif agg_func == 'sum':
            return torch.sum
        elif agg_func == 'mean':
            return torch.mean
        else:
            raise NotImplementedError()
    return agg_func

class GraphVertExtraLinModel(nn.Module):
    def __init__(self, g_feature_n, g_feature_out_n=None,
                 resnet=True, 
                 int_d = None, layer_n = None, 
                 init_noise=1e-5, agg_func=None, GS=1, 
                 combine_in = False, 
                 OUT_DIM=1,force_lin_init=False, 
                 use_highway = False, 
                 use_graph_conv = True, 
                 extra_lin_int_d = 128 ):
        super(GraphVertExtraLinModel, self).__init__()

        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n
        print("g_feature_n=", g_feature_n)
        self.use_highway = use_highway
        if use_highway:
            self.gml = GraphMatHighwayLayers(g_feature_n, g_feature_out_n, 
                                      resnet=resnet, noise=init_noise, 
                                      agg_func=parse_agg_func(agg_func), 
                                      GS=GS)
        else:
            self.gml = GraphMatLayers(g_feature_n, g_feature_out_n, 
                                      resnet=resnet, noise=init_noise, 
                                      agg_func=parse_agg_func(agg_func), 
                                      GS=GS)

        self.combine_in = combine_in
        self.use_graph_conv = use_graph_conv
        lin_layer_feat = 0
        if use_graph_conv:
            lin_layer_feat += g_feature_out_n[-1]

        if combine_in :
            lin_layer_feat += g_feature_n
        if self.use_highway:
            lin_layer_feat += np.sum(g_feature_out_n)
            
        self.lin_out1 = nn.Linear(lin_layer_feat, extra_lin_int_d)
        self.lin_out2 = nn.Linear(extra_lin_int_d, OUT_DIM)

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, init_noise)
                    nn.init.constant_(m.bias, 0)

    def forward(self, args):
        G = args[0]
        x_G = args[1]
        if self.use_highway:
            G_features, G_highway = self.gml(G, x_G)
            G_highway_flatten = G_highway.reshape(G_highway.shape[0], 
                                                  G_highway.shape[1], -1)
        else:
            G_features = self.gml(G, x_G)
        
        g_squeeze = G_features.squeeze(1)
        out_feat = []
        if self.use_graph_conv :
            out_feat.append(g_squeeze)
        if self.combine_in:
            out_feat.append(x_G)
        if self.use_highway:
            out_feat.append(G_highway_flatten)

        lin_input = torch.cat(out_feat, -1)
            
        x_1 = self.lin_out1(lin_input)
        x_2 = self.lin_out2(F.relu(x_1))
        
        return x_2


class MSELogNormalLoss(nn.Module):
    def __init__(self, use_std_term = True, 
                 use_log1p=True, std_regularize=0.0, 
                 std_pow = 2.0):
        super(MSELogNormalLoss, self).__init__()
        self.use_std_term = use_std_term
        self.use_log1p = use_log1p
        self.std_regularize = std_regularize
        self.std_pow = std_pow

    def __call__(self, y, mu, std):
        if self.use_log1p:
            log = torch.log1p
        else:
            log = torch.log
        std = std + self.std_regularize

        std_term = -0.5 * log(2*np.pi * std**2 ) 
        log_pdf = - (y-mu)**2/(2.0 * std **self.std_pow)
        if self.use_std_term :
            log_pdf += std_term 
        return -log_pdf.mean()


def log_normal_nolog(y, mu, std):
    element_wise =  -(y - mu)**2 / (2*std**2)  - std
    return element_wise

def log_student_t(y, mu, std, v=1.0):
    return -torch.log(1.0 + (y-mu)**2/(v * std)) - std

def log_normal(y, mu, std):
    element_wise =  -(y - mu)**2 / (2*std**2)  - torch.log(std)
    return element_wise

class MSECustomLoss(nn.Module):
    def __init__(self, use_std_term = True, 
                 use_log1p=True, std_regularize=0.0, 
                 std_pow = 2.0):
        super(MSECustomLoss, self).__init__()
        self.use_std_term = use_std_term
        self.use_log1p = use_log1p
        self.std_regularize = std_regularize
        self.std_pow = std_pow

    def __call__(self, y, mu, std):
        if self.use_log1p:
            log = torch.log1p
        else:
            log = torch.log
        std = std + self.std_regularize
        return -log_normal(y, mu, std).mean()

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mseloss = nn.MSELoss()
        
    def __call__(self, y, x, mask):
        x_masked = x[mask>0].reshape(-1, 1)
        y_masked = y[mask>0].reshape(-1, 1)
        return self.mseloss(x_masked, y_masked)

class MaskedMSSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSSELoss, self).__init__()
        
    def __call__(self, y, x, mask):
        x_masked = x[mask>0].reshape(-1, 1)
        y_masked = y[mask>0].reshape(-1, 1)
        return ((x_masked - y_masked)**4).mean()

class MaskedMSEScaledLoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mseloss = nn.MSELoss()
        
    def __call__(self, y, x, mask):
        x_masked = x[mask>0].reshape(-1, 1)
        y_masked = y[mask>0].reshape(-1, 1)
        return self.mseloss(x_masked, y_masked)


class NormUncertainLoss(nn.Module):
    def __init__(self, 
                 mu_scale = torch.Tensor([1.0]), 
                 std_scale = torch.Tensor([1.0]), 
                 use_std_term = True, 
                 use_log1p=False, std_regularize=0.0, 
                 std_pow = 2.0):
        super(NormUncertainLoss, self).__init__()
        self.use_std_term = use_std_term
        self.use_log1p = use_log1p
        self.std_regularize = std_regularize
        self.std_pow = std_pow
        self.mu_scale = mu_scale
        self.std_scale = std_scale

    def __call__(self, pred, y,  mask):
        # Force FP32 precision for numerical stability under AMP autocast
        mu = pred['mu'].float() 
        std = pred['std'].float()
        y = y.float()
        mask = mask.float()

        if self.use_log1p:
            log = torch.log1p
        else:
            log = torch.log
        std = std + self.std_regularize

        y_scaled = y / self.mu_scale
        mu_scaled = mu / self.mu_scale
        std_scaled = std / self.std_scale 

        y_scaled_masked = y_scaled[mask>0].reshape(-1, 1)
        mu_scaled_masked = mu_scaled[mask>0].reshape(-1, 1)
        std_scaled_masked = std_scaled[mask>0].reshape(-1, 1)

        return -log_normal_nolog(y_scaled_masked, 
                              mu_scaled_masked, 
                              std_scaled_masked).mean()

class NormUncertainCalibrationLoss(nn.Module):
    """
    Masked uncertainty loss with per-molecule calibration shift (c_m)
    and L1 or L2 penalty (in ppm) to anchor the absolute chemical shift scale.
    """
    def __init__(self, 
                 mu_scale = torch.Tensor([1.0]), 
                 std_scale = torch.Tensor([1.0]), 
                 std_regularize=0.0,
                 lambda_penalty=1.0,
                 cm_norm="L2"):
        super(NormUncertainCalibrationLoss, self).__init__()
        self.mu_scale = mu_scale
        self.std_scale = std_scale
        self.std_regularize = std_regularize
        self.lambda_penalty = lambda_penalty
        self.cm_norm = cm_norm.upper()

    def forward(self, pred, y, mask):
        # Force FP32 precision for numerical stability under AMP autocast
        mu  = pred['mu'].float() 
        std = pred['std'].float()
        y = y.float()
        mask = mask.float()
        
        std = std + self.std_regularize

        mu_scale = self.mu_scale.to(y.device).float()
        std_scale = self.std_scale.to(y.device).float()

        y_scaled = y / mu_scale
        mu_scaled = mu / mu_scale
        std_scaled = std / std_scale

        variance = std_scaled ** 2
        # Detach variance in inverse-variance weights w so the network does not inflate std to reduce c_m penalty
        w = mask / variance.detach()
        delta = y_scaled - mu_scaled
        
        sum_w = w.sum(dim=1, keepdim=True)
        sum_w_clamped = torch.clamp(sum_w, min=1e-9)
        
        # Inverse-variance weighted per-molecule shift c_m in mu_scale units (with gradients wrt mu enabled)
        c_m = (delta * w).sum(dim=1, keepdim=True) / sum_w_clamped
        c_m = c_m * (sum_w > 0).float()
        
        # Use detached c_m when evaluating the NLL term
        mu_calib = mu_scaled + c_m.detach()
        
        # Element-wise pseudo-NLL (without log, matching the original log_normal_nolog formulation)
        element_wise = - (y_scaled - mu_calib)**2 / (2 * variance) - std_scaled
        
        log_pdf = element_wise * mask
        nll_loss = -log_pdf.sum() / torch.clamp(mask.sum(), min=1.0)
        
        # Convert c_m back to physical units (ppm) for transparent L1/L2 regularization
        c_m_ppm = c_m * mu_scale
        active_mols = (sum_w > 0).float()
        if self.cm_norm == "L1":
            c_m_penalty = c_m_ppm.abs().sum() / torch.clamp(active_mols.sum(), min=1.0)
        else:
            c_m_penalty = (c_m_ppm ** 2).sum() / torch.clamp(active_mols.sum(), min=1.0)
        
        total_loss = nll_loss + self.lambda_penalty * c_m_penalty
        
        return total_loss, nll_loss, c_m_penalty


_LOG_2PI = float(np.log(2.0 * np.pi))


class BetaNLLLoss(nn.Module):
    """
    Masked heteroscedastic Gaussian beta-NLL loss (Seitzer et al., ICLR 2022):
        L_{beta-NLL} = sg[var^beta] * 0.5 * ((y - mu)^2 / var + log(var) + log(2*pi))

    Supports both single-head models ({'mu', 'var'}) and multi-head bootstrap ensembles
    ({'head_mu', 'head_var', 'head_mask', 'is_training'}):
      - During training with multi-head outputs, computes beta-NLL independently on each
        bootstrap-selected head (preventing head co-adaptation while preserving unbiased
        aleatoric variance per head and epistemic diversity across heads).
      - During evaluation (is_training=False or beta==0.0 without head training), computes
        exact unweighted Gaussian NLL on the moment-matched ensemble (mu, var_alea + var_epi).

    Returns a 3-tuple (train_loss, true_unweighted_nll, zero_penalty) compatible with run_epoch.
    """
    def __init__(self, beta=0.5, min_var=1e-4, include_const=True):
        super(BetaNLLLoss, self).__init__()
        if not (0.0 <= float(beta) <= 1.0):
            raise ValueError(f"beta must be in [0.0, 1.0], got {beta}")
        self.beta = float(beta)
        self.min_var = float(min_var)
        self.include_const = bool(include_const)

    def _compute_loss(self, mu, var, y, eff_mask):
        var = torch.clamp(var, min=self.min_var)
        nll_elem = 0.5 * ((y - mu) ** 2 / var + torch.log(var))
        if self.include_const:
            nll_elem = nll_elem + 0.5 * _LOG_2PI

        denom = torch.clamp(eff_mask.sum(), min=1.0)
        true_nll = (nll_elem * eff_mask).sum() / denom

        if self.beta > 0.0:
            weighted_elem = nll_elem * (var.detach() ** self.beta)
            train_loss = (weighted_elem * eff_mask).sum() / denom
        else:
            train_loss = true_nll
        return train_loss, true_nll

    def forward(self, pred, y, mask):
        # Force FP32 precision for numerical stability under AMP autocast
        y = y.float()
        mask = mask.float()

        if pred.get('is_training', False) and ('head_mu' in pred) and ('head_var' in pred):
            head_mu = pred['head_mu'].float()       # (K, B, N, D)
            head_var = pred['head_var'].float()     # (K, B, N, D)
            head_mask = pred['head_mask'].float()   # (K, B)
            eff_mask = mask.unsqueeze(0) * head_mask[:, :, None, None]
            train_loss, true_nll = self._compute_loss(head_mu, head_var, y.unsqueeze(0), eff_mask)
            zero_pen = torch.zeros((), device=head_mu.device, dtype=head_mu.dtype)
            return train_loss, true_nll, zero_pen

        mu = pred['mu'].float()
        if 'var' in pred:
            var = pred['var'].float()
        else:
            var = pred['std'].float() ** 2

        train_loss, true_nll = self._compute_loss(mu, var, y, mask)
        zero_pen = torch.zeros((), device=mu.device, dtype=mu.dtype)
        return train_loss, true_nll, zero_pen


# ==============================================================================
# NMRResPredictV2: Lightweight Williams & Jonas (2023) Polynomial GNN
#                  + 3D Spatial RBF Message Passing
#                  + Multi-Head Bootstrap Ensemble with Heteroscedastic beta-NLL
# ==============================================================================

class MaskedBatchNorm1d(nn.Module):
    """BatchNorm1d applied strictly to non-padded atoms (input_mask > 0)."""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(MaskedBatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    def forward(self, x, mask):
        # x: (B, N, F), mask: (B, N) bool or float
        orig_dtype = x.dtype
        m = mask > 0.5
        out = torch.zeros_like(x, dtype=torch.float32)
        valid = x[m].float()
        if valid.shape[0] > 1 or not self.training:
            out[m] = self.bn(valid)
        elif valid.shape[0] == 1:
            # Fallback to running stats if batch has only 1 valid atom
            was_training = self.bn.training
            self.bn.eval()
            out[m] = self.bn(valid)
            self.bn.train(was_training)
        return out.to(orig_dtype)


class PolyGraphTerm(nn.Module):
    """
    2-layer MLP per bond channel followed by multiplication with a precomputed
    polynomial adjacency term (G^1, G^2, or diag(G^3)), normalized by neighbor count.
    Matches Williams & Jonas (Chem. Sci., 2023) GraphMatLayerFastPow2, but avoids
    recomputing matrix powers at every layer.
    """
    def __init__(self, in_dim, out_dim, GS=5, is_diag=False, dropout=0.01):
        super(PolyGraphTerm, self).__init__()
        self.GS = GS
        self.is_diag = is_diag
        self.lin1 = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(GS)])
        self.lin2 = nn.ModuleList([nn.Linear(out_dim, out_dim) for _ in range(GS)])
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, G_term, G_deg, x):
        # x: (B, N, C)
        # multi_x: (B, GS, N, P)
        multi_x = torch.stack(
            [self.dropout(self.lin2[i](F.relu(self.lin1[i](x)))) for i in range(self.GS)],
            dim=1
        )
        if self.is_diag:
            # G_term: (B, GS, N) diagonal of G^3
            xout = G_term.unsqueeze(-1) * multi_x
        else:
            # G_term: (B, GS, N, N)
            xout = torch.matmul(G_term, multi_x)
        # Normalize by topological neighbor count per bond channel
        xout = xout / G_deg.unsqueeze(-1)
        return xout  # (B, GS, N, P)


class PolyGraphSpatialLayer(nn.Module):
    """
    Single message-passing block combining:
      1) Topological polynomial GNN terms: G^1 (1-hop), G^2 (2-hop), diag(G^3) (3-cycles)
         aggregated across bond channels via goodmax (Williams & Jonas 2023);
      2) Explicit G^0 self-feature update (self_lin(x)) so leaf-node identity features
         (such as CH2 proton differentiation flags) interact non-linearly with incoming
         neighborhood context at every layer even in pure 2D (--no_3d);
      3) Optional lightweight 3D SE(3)-invariant spatial RBF + 1/r^3 message passing
         to capture stereochemistry, through-space ring currents, and CH2 diastereotopicity;
      4) ReLU + LayerNorm + residual connection.
    """
    def __init__(self, in_dim, out_dim, GS=5, dropout=0.01, spatial_dim=0, spatial_hidden=64):
        super(PolyGraphSpatialLayer, self).__init__()
        self.term_g1 = PolyGraphTerm(in_dim, out_dim, GS=GS, is_diag=False, dropout=dropout)
        self.term_g2 = PolyGraphTerm(in_dim, out_dim, GS=GS, is_diag=False, dropout=dropout)
        self.term_g3d = PolyGraphTerm(in_dim, out_dim, GS=GS, is_diag=True, dropout=dropout)
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.self_drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.spatial_dim = int(spatial_dim)
        if self.spatial_dim > 0:
            self.spatial_in = nn.Linear(in_dim, spatial_hidden)
            self.spatial_out = nn.Linear(self.spatial_dim * spatial_hidden, out_dim)
            self.spatial_drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.norm = nn.LayerNorm(out_dim)
        self.use_res = (in_dim == out_dim)

    def forward(self, G1, G2, G3_diag, G_deg, x, spatial_mat=None, spatial_deg=None, mask=None):
        # Sum polynomial terms, then take max across GS bond channels
        topo = (
            self.term_g1(G1, G_deg, x)
            + self.term_g2(G2, G_deg, x)
            + self.term_g3d(G3_diag, G_deg, x)
        )
        xout = torch.max(topo, dim=1)[0] + self.self_drop(self.self_lin(x))  # (B, N, P)

        if self.spatial_dim > 0 and spatial_mat is not None:
            # spatial_mat: (B, N, N, S), x_sp: (B, N, H)
            x_sp = F.relu(self.spatial_in(x))
            agg_sp = torch.einsum("bnms,bmd->bnsd", spatial_mat, x_sp)
            agg_sp = agg_sp / spatial_deg.unsqueeze(-1)
            agg_sp = agg_sp.reshape(x.shape[0], x.shape[1], -1)
            xout = xout + self.spatial_drop(self.spatial_out(agg_sp))

        xout = self.norm(F.relu(xout))
        if self.use_res:
            xout = xout + x
        if mask is not None:
            xout = xout * mask.unsqueeze(-1).to(xout.dtype)
        return xout


class _LNResBlock(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super(_LNResBlock, self).__init__()
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        return self.norm(x + self.drop(self.l2(F.relu(self.l1(x)))))


class ProbabilisticResNetHead(nn.Module):
    """
    Lightweight residual readout head outputting both chemical shift mean (mu)
    and strictly positive physical variance (var in ppm^2) via softplus + offset + clamp
    (following Seitzer et al., ICLR 2022 / martius-lab/beta-nll).
    """
    def __init__(self, in_dim, hidden_dim=128, depth=3, out_dim=1,
                 init_var=25.0, min_var=0.01, max_var=10000.0, dropout=0.0):
        super(ProbabilisticResNetHead, self).__init__()
        self.min_var = float(min_var)
        self.max_var = float(max_var)
        delta_var = float(init_var) - self.min_var
        self.init_var_offset = (
            float(delta_var) if delta_var > 20.0 else float(np.log(np.expm1(delta_var)))
        )

        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([_LNResBlock(hidden_dim, dropout=dropout) for _ in range(depth)])
        self.mu_out = nn.Linear(hidden_dim, out_dim)

        # Dedicated variance projection branch to avoid gradient interference with mu_out
        var_h = max(hidden_dim // 2, 32)
        self.var_fc1 = nn.Linear(hidden_dim, var_h)
        self.var_fc2 = nn.Linear(var_h, out_dim)

    def forward(self, x):
        h = self.norm_in(F.relu(self.lin_in(x)))
        for blk in self.blocks:
            h = blk(h)
        mu = self.mu_out(h).float()
        raw_var = self.var_fc2(F.relu(self.var_fc1(h))).float()
        var = F.softplus(raw_var + self.init_var_offset) + self.min_var
        var = torch.clamp(var, min=self.min_var, max=self.max_var)
        return mu, var


class NMRResPredictV2(nn.Module):
    """
    Lightweight, rich-feature NMR chemical shift and full-uncertainty GNN combining:
      - Williams & Jonas (Chem. Sci. 2023) polynomial GNN encoder (int_d=256, layer_n=8,
        G^1 + G^2 + diag(G^3), MaskedBatchNorm1d, LayerNorm, Xavier initialization);
      - Lightweight 3D SE(3)-invariant spatial RBF + 1/r^3 message passing;
      - Multi-head bootstrap ensemble (mixture_n heads) where each head predicts BOTH
        mean (mu_k) and aleatoric variance (var_k in ppm^2) trained with beta-NLL;
      - Exact Law of Total Variance uncertainty decomposition at inference:
            var_total = mean_k(var_k) [aleatoric] + var_k(mu_k) [epistemic].
    """
    accepts_spatial_and_idx = True

    def __init__(
        self,
        g_feature_n,
        int_d=256,
        layer_n=8,
        GS=5,
        OUT_DIM=1,
        spatial_dim=6,
        spatial_hidden=64,
        dropout=0.01,
        resnet_d=128,
        resnet_depth=3,
        mixture_n=5,
        mixture_num_obs_per=4,
        init_var=25.0,
        min_var=0.01,
        max_var=10000.0,
        **kwargs,
    ):
        super(NMRResPredictV2, self).__init__()
        self.g_feature_n = int(g_feature_n)
        self.int_d = int(int_d)
        self.layer_n = int(layer_n)
        self.GS = int(GS)
        self.OUT_DIM = int(OUT_DIM)
        self.spatial_dim = int(spatial_dim)
        self.mixture_n = int(mixture_n)
        self.mixture_num_obs_per = int(min(mixture_num_obs_per, mixture_n))
        self.min_var = float(min_var)
        self.max_var = float(max_var)

        self.input_norm = MaskedBatchNorm1d(self.g_feature_n)

        layers = []
        for li in range(self.layer_n):
            in_c = self.g_feature_n if li == 0 else self.int_d
            layers.append(
                PolyGraphSpatialLayer(
                    in_dim=in_c,
                    out_dim=self.int_d,
                    GS=self.GS,
                    dropout=dropout,
                    spatial_dim=self.spatial_dim,
                    spatial_hidden=spatial_hidden,
                )
            )
        self.layers = nn.ModuleList(layers)

        self.heads = nn.ModuleList([
            ProbabilisticResNetHead(
                in_dim=self.int_d,
                hidden_dim=resnet_d,
                depth=resnet_depth,
                out_dim=self.OUT_DIM,
                init_var=init_var,
                min_var=min_var,
                max_var=max_var,
                dropout=dropout,
            )
            for _ in range(self.mixture_n)
        ])

        # Standard Xavier uniform initialization (matching Williams & Jonas 2023)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        # Keep variance head final layer near zero at step 0 so initial variance == init_var
        for h in self.heads:
            nn.init.normal_(h.var_fc2.weight, 0.0, 1e-3)
            nn.init.constant_(h.var_fc2.bias, 0.0)

    def _build_head_mask(self, batch_size, input_idx, device):
        K = self.mixture_n
        m = self.mixture_num_obs_per
        if K <= 1 or m >= K or input_idx is None:
            return torch.ones((K, batch_size), device=device, dtype=torch.float32)

        mask_np = np.zeros((K, batch_size), dtype=np.float32)
        idx_list = input_idx.detach().cpu().numpy().tolist() if torch.is_tensor(input_idx) else list(input_idx)
        for b, raw_i in enumerate(idx_list):
            seed = int(raw_i) % (2 ** 31 - 1)
            chosen = np.random.RandomState(seed).permutation(K)[:m]
            mask_np[chosen, b] = 1.0
        return torch.from_numpy(mask_np).to(device)

    def forward(self, args):
        if len(args) == 4:
            G, vect_feat, mat_feat, input_idx = args
        elif len(args) == 3:
            G, vect_feat, mat_feat = args
            input_idx = None
        else:
            G, vect_feat = args
            mat_feat = None
            input_idx = None

        # Derive atom presence mask (B, N) from non-zero atom feature vectors
        atom_mask = (vect_feat.abs().sum(dim=-1) > 0).float()

        x = self.input_norm(vect_feat, atom_mask)

        # Precompute polynomial adjacency powers ONCE per batch across all layers
        G1 = G                                       # (B, GS, N, N)
        G2 = torch.matmul(G1, G1)                    # (B, GS, N, N)
        G3_diag = (G2 * G1).sum(dim=-1)              # (B, GS, N) exact diag(G^3) for symmetric G
        G_deg = torch.clamp(G1.sum(dim=-1), min=1.0) # (B, GS, N)

        spatial_mat = None
        spatial_deg = None
        if self.spatial_dim > 0 and mat_feat is not None and mat_feat.shape[-1] == self.spatial_dim:
            spatial_mat = mat_feat
            spatial_deg = torch.clamp(spatial_mat.sum(dim=2), min=1.0)  # (B, N, S)

        for layer in self.layers:
            x = layer(G1, G2, G3_diag, G_deg, x, spatial_mat=spatial_mat, spatial_deg=spatial_deg, mask=atom_mask)

        head_mus = []
        head_vars = []
        for head in self.heads:
            mu_k, var_k = head(x)
            head_mus.append(mu_k)
            head_vars.append(var_k)

        head_mu = torch.stack(head_mus, dim=0)    # (K, B, N, D)
        head_var = torch.stack(head_vars, dim=0)  # (K, B, N, D)

        # Law of Total Variance (Gaussian Mixture moment matching)
        mu = head_mu.mean(dim=0)
        var_alea = head_var.mean(dim=0)
        if self.mixture_n > 1:
            var_epi = ((head_mu - mu.unsqueeze(0)) ** 2).mean(dim=0)
        else:
            var_epi = torch.zeros_like(mu)

        var_total = torch.clamp(var_alea + var_epi, min=self.min_var, max=self.max_var)
        std_total = torch.sqrt(var_total)
        std_alea = torch.sqrt(torch.clamp(var_alea, min=self.min_var))
        std_epi = torch.sqrt(torch.clamp(var_epi, min=0.0) + 1e-12)

        ret = {
            'mu': mu,
            'var': var_total,
            'std': std_total,
            'std_alea': std_alea,
            'std_epi': std_epi,
            'is_training': self.training,
        }
        if self.training:
            ret['head_mu'] = head_mu
            ret['head_var'] = head_var
            ret['head_mask'] = self._build_head_mask(x.shape[0], input_idx, x.device)
        return ret
