import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculate_laplacian(adj):  # 计算A~
    D = np.diag(np.ravel(adj.sum(axis=0)) ** (-0.5))
    adj = np.dot(D, np.dot(adj, D))
    adj = torch.tensor(adj, dtype=torch.float32)
    return adj.to(device)


class GraphConvolution(nn.Module):  # 图卷积层
    def __init__(
            self,
            units,
            feature_units,
            A,
            activation=nn.ReLU(),
            bias_initializer=None,
            kernel_initializer=nn.Linear,
            use_bias=True,
            bias=True,
    ):
        super(GraphConvolution, self).__init__()
        self.units = units
        # self.adj = calculate_laplacian(A)
        self.adj = A
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.use_bias = use_bias
        self.bias = bias
        self.kernel = kernel_initializer(100, units)
        # old is self.featrue = kernel_initializer(feature_units, 100)
        self.feature = kernel_initializer(feature_units, 100)

    def init_weights(self):
        self.kernel.bias.data.zero_()
        self.kernel.weight.data.uniform_(-0.1, 0.1)

    def build(self, input_shapes):
        _batch_dim, n_nodes, features = input_shapes.shape  # 传递数组的形状
        self.kernel = self.kernel_initializer(features, features).to(device)
        self.init_weights()
        if self.adj is not None:
            adj_init = self.adj
        else:
            adj_init = torch.zeros((n_nodes, n_nodes)).to(device)

        if self.use_bias:
            self.bias = self.bias_initializer
        else:
            self.bias = None

    def forward(self, x):
        # features_size  [B, N, T ,F]
        # [B,N,T,F]
        x = self.feature(x)
        # [B T F N]
        node_last = x.permute(0, 2, 3, 1)
        # [B T F N]
        neighbors = torch.matmul(node_last, self.adj.to(x.device))
        # [B,N,T,F]
        h_graph = neighbors.permute(0, 3, 1, 2)
        output = self.kernel(h_graph)  # [B,N,T,F]
        if self.bias:
            output = output + self.bias

        output = self.activation(output)
        return output


class GCN_FIGURE(nn.Module):
    def __init__(
            self,
            adj,
            gc_layer_sizes,
            feature_sizes,
            gc_activations=None,
            variates=1,
            bias_initializer=None
    ):
        super(GCN_FIGURE, self).__init__()

        self.adj = adj
        self.gc_layer_sizes = gc_layer_sizes
        self.feature_sizes = feature_sizes
        self.gc_activations = gc_activations
        self.n_gc_layers = len(gc_layer_sizes)
        self.variates = variates
        self.bias_initializer = bias_initializer

        self._gc_layers = nn.ModuleList([
            GraphConvolution(
                units=self.variates * layer_size,
                feature_units=feature_size,
                A=self.adj,
                activation=activation,
                bias_initializer=self.bias_initializer,
                kernel_initializer=nn.Linear,
            )
            for layer_size, feature_size, activation in
            zip(self.gc_layer_sizes, self.feature_sizes, self.gc_activations)
        ])

    def forward(self, x):
        # x  [B,N,T,F]
        for layer in self._gc_layers:
            x = layer(x)
        # B, N, T, F
        return x


class GraphConvolution_BASELINE(nn.Module):  # 图卷积层
    def __init__(
            self,
            units,
            feature_units,
            A,
            activation=nn.ReLU(),
            bias_initializer=None,
            kernel_initializer=nn.Linear,
            use_bias=True,
            bias=True,
    ):
        super(GraphConvolution_BASELINE, self).__init__()
        self.units = units
        self.adj = A
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.use_bias = use_bias
        self.bias = bias
        self.kernel = kernel_initializer(feature_units, units)

    def init_weights(self):
        self.kernel.bias.data.zero_()
        self.kernel.weight.data.uniform_(-0.1, 0.1)

    def build(self, input_shapes):
        _batch_dim, n_nodes, features = input_shapes.shape  # 传递数组的形状
        self.kernel = self.kernel_initializer(features, features).to(device)
        self.init_weights()
        if self.adj is not None:
            adj_init = self.adj
        else:
            adj_init = torch.zeros((n_nodes, n_nodes)).to(device)

        if self.use_bias:
            self.bias = self.bias_initializer
        else:
            self.bias = None

    def forward(self, x):
        # features_size  [B, N, F]
        # [B,N,F]
        node_last = x.permute(0, 2, 1)
        # [B F N]
        neighbors = torch.matmul(node_last, self.adj.to(x.device))
        # [B,N,F]
        h_graph = neighbors.permute(0, 2, 1)
        output = self.kernel(h_graph)  # [B,N,F]
        if self.bias:
            output = output + self.bias

        output = self.activation(output)
        return output


class GCN_BASELINE(nn.Module):
    def __init__(
            self,
            adj,
            gc_layer_sizes,
            feature_sizes,
            gc_activations=None,
            variates=1,
            bias_initializer=None
    ):
        super(GCN_BASELINE, self).__init__()

        self.adj = torch.tensor(adj)
        self.gc_layer_sizes = gc_layer_sizes
        self.feature_sizes = feature_sizes
        self.gc_activations = gc_activations
        self.n_gc_layers = len(gc_layer_sizes)
        self.variates = variates
        self.bias_initializer = bias_initializer

        self._gc_layers = nn.ModuleList([
            GraphConvolution_BASELINE(
                units=self.variates * layer_size,
                feature_units=feature_size,
                A=self.adj,
                activation=activation,
                bias_initializer=self.bias_initializer,
                kernel_initializer=nn.Linear,
            )
            for layer_size, feature_size, activation in
            zip(self.gc_layer_sizes, self.feature_sizes, self.gc_activations)
        ])

    def forward(self, x):
        # x  [B,N,T,F]
        for layer in self._gc_layers:
            x = layer(x)
        # B, N, T, F
        return x
