import torch
import torch.nn as nn
import numpy as np


def calculate_laplacian(adj):  # 计算A~
    D = np.diag(np.ravel(adj.sum(axis=0)) ** (-0.5))
    adj = np.dot(D, np.dot(adj, D))
    return adj


class FixedAdjacencyGraphConvolution(nn.Module):  # 图卷积层
    """
    Graph Convolution (GCN) Keras layer.
    The implementation is based on https://github.com/tkipf/keras-gcn.
    Original paper: Semi-Supervised Classification with Graph Convolutional Networks. Thomas N. Kipf, Max Welling,
    International Conference on Learning Representations (ICLR), 2017 https://github.com/tkipf/gcn
    Notes:
      - The inputs are 3 dimensional tensors: batch size, sequence length, and number of nodes.
      - This class assumes that a simple unweighted or weighted adjacency matrix is passed to it,
        the normalized Laplacian matrix is calculated within the class.
    Args:
        units (int): dimensionality of output feature vectors
        A (N x N): weighted/unweighted adjacency matrix
        activation (str or func): nonlinear activation applied to layer's output to obtain output features
        use_bias (bool): toggles an optional bias
        kernel_initializer (str or func, optional): The initialiser to use for the weights.
        kernel_regularizer (str or func, optional): The regulariser to use for the weights. 有序化
        kernel_constraint (str or func, optional): The constraint to use for the weights.  限制
        bias_initializer (str or func, optional): The initialiser to use for the bias.
        bias_regularizer (str or func, optional): The regulariser to use for the bias.
        bias_constraint (str or func, optional): The constraint to use for the bias.
    """

    def __init__(
            self,
            units,
            A,
            activation=None,
            use_bias=True,
            input_dim=None,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=None,
            kernel_constraint=None,
            bias_initializer="zeros",
            bias_regularizer=None,
            bias_constraint=None,
            **kwargs,
    ):
        super(FixedAdjacencyGraphConvolution, self).__init__()
        if "input_shape" not in kwargs and input_dim is not None:
            kwargs["input_shape"] = (input_dim,)

        self.units = units
        self.adj = calculate_laplacian(A) if A is not None else None

        self.activation = activation
        self.use_bias = use_bias

        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint

        super().__init__(**kwargs)

    def get_config(self):
        """
        Gets class configuration for Keras serialization.
        Used by Keras model serialization.
        Returns:
            A dictionary that contains the config of the layer
        """

        config = {
            "units": self.units,
            "use_bias": self.use_bias,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "kernel_constraint": self.kernel_constraint,
            "bias_initializer": self.bias_initializer,
            "bias_regularizer": self.bias_regularizer,
            "bias_constraint": self.bias_constraint,
            # the adjacency matrix argument is required, but
            # (semi-secretly) supports None for loading from a saved
            # model, where the adjacency matrix is a saved weight
            "A": None,
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        """
        Computes the output shape of the layer.
        Assumes the following inputs:
        Args:
            input_shapes (tuple of int)
                Shape tuples can include None for free dimensions, instead of an integer.
        Returns:
            An input shape tuple.
        """
        feature_shape = input_shapes

        return feature_shape[0], feature_shape[1], self.units

    def build(self, input_shapes):
        """
        Builds the layer
        Args:
            input_shapes (list of int): shapes of the layer's inputs (the batches of node features)
        """
        _batch_dim, n_nodes, features = input_shapes

        if self.adj is not None:
            adj_init = self.adj
        else:
            adj_init = torch.zeros((n_nodes, n_nodes))

        self.A = adj_init
        self.kernel = self.kernel_initializer

        if self.use_bias:
            self.bias = self.bias_initializer
        else:
            self.bias = None
        self.built = True

    def call(self, features):
        """
        Applies the layer.
        Args:
            features (ndarray): node features (size B x N x F), where B is the batch size, F = TV is
                the feature size (consisting of the sequence length and the number of variates), and
                N is the number of nodes in the graph.
        Returns:
            Keras Tensor that represents the output of the layer.
        """

        # Calculate the layer operation of GCN
        # shape = B x F x N
        nodes_last = features.transpose(0, 2, 1)
        neighbours = torch.matmul(nodes_last, self.A)
        print('neighbors:', neighbours.shape)
        # shape = B x N x F
        h_graph = neighbours.transpose(0, 2, 1)
        # shape = B x N x units
        output = torch.matmul(h_graph, self.kernel)

        # Add optional bias & apply activation
        if self.bias is not None:
            output += self.bias

        output = self.activation(output)

        return output


class GCN_LSTM(nn.Module):
    def __init__(
            self,
            seq_len,
            adj,
            gc_layer_sizes,
            lstm_layer_sizes,
            forecast_horizon,
            gc_activations=None,
            generator=None,
            lstm_activations=None,
            bias=True,
            dropout=0.5,
            kernel_initializer=None,
            kernel_regularizer=None,
            kernel_constraint=None,
            bias_initializer=None,
            bias_regularizer=None,
            bias_constraint=None,
    ):
        if generator is not None:
            if not isinstance(generator, SlidingFeaturesNodeGenerator):
                raise ValueError(
                    f"generator: expected a SlidingFeaturesNodeGenerator, found {type(generator).__name__}"
                )

            if seq_len is not None or adj is not None:
                raise ValueError(
                    "expected only one of generator and (seq_len, adj) to be specified, found multiple"
                )

            adj = generator.graph.to_adjacency_matrix(weighted=True).todense()
            seq_len = generator.window_size
            variates = generator.variates
        else:
            variates = None

        super(GCN_LSTM, self).__init__()

        n_gc_layers = len(gc_layer_sizes)
        n_lstm_layers = len(lstm_layer_sizes)

        self.lstm_layer_sizes = lstm_layer_sizes
        self.gc_layer_sizes = gc_layer_sizes
        self.bias = bias
        self.dropout = dropout
        self.adj = adj
        self.n_nodes = adj.shape[0]
        self.forecast_horizon = forecast_horizon
        self.n_features = seq_len
        self.seq_len = seq_len
        self.multivariate_input = variates is not None
        self.variates = variates if self.multivariate_input else 1
        self.outputs = self.n_nodes * self.variates

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        # Activation function for each gcn layer
        if gc_activations is None:
            gc_activations = ["relu"] * n_gc_layers
        elif len(gc_activations) != n_gc_layers:
            raise ValueError(
                "Invalid number of activations; require one function per graph convolution layer"
            )
        self.gc_activations = gc_activations

        # Activation function for each lstm layer
        if lstm_activations is None:
            lstm_activations = ["tanh"] * n_lstm_layers
        elif len(lstm_activations) != n_lstm_layers:
            padding_size = n_lstm_layers - len(lstm_activations)
            if padding_size > 0:
                lstm_activations = lstm_activations + ["tanh"] * padding_size
            else:
                raise ValueError(
                    "Invalid number of activations; require one function per lstm layer"
                )
        self.lstm_activations = lstm_activations

        self._gc_layers = [
            FixedAdjacencyGraphConvolution(
                units=self.variates * layer_size,
                A=self.adj,
                activation=activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_initializer=self.bias_initializer,
                bias_regularizer=self.bias_regularizer,
                bias_constraint=self.bias_constraint,
            )
            for layer_size, activation in zip(self.gc_layer_sizes, self.gc_activations)
        ]
        self._lstm_layers = [
            LSTM(layer_size, activation=activation, return_sequences=True)
            for layer_size, activation in zip(
                self.lstm_layer_sizes[:-1], self.lstm_activations
            )
        ]
        self._lstm_layers.append(
            LSTM(
                self.lstm_layer_sizes[-1],
                activation=self.lstm_activations[-1],
                return_sequences=False  # ,
                # dropout=0.1
            )
        )
        self._bn = BatchNormalization(),
        self._decoder_layer = Dense(self.n_nodes * self.forecast_horizon)

    def __call__(self, x):

        x_in, out_indices = x

        h_layer = x_in
        if not self.multivariate_input:
            # normalize to always have a final variate dimension, with V = 1 if it doesn't exist
            # shape = B x N x T x 1
            h_layer = tf.expand_dims(h_layer, axis=-1)

        # flatten variates into sequences, for convolution
        # shape B x N x (TV)
        h_layer = Reshape((self.n_nodes, self.seq_len * self.variates))(h_layer)

        for layer in self._gc_layers:
            h_layer = layer(h_layer)

        # return the layer to its natural multivariate tensor form
        # shape B x N x T' x V (where T' is the sequence length of the last GC)
        h_layer = Reshape((self.n_nodes, -1, self.variates))(h_layer)
        # put time dimension first for LSTM layers
        # shape B x T' x N x V
        h_layer = Permute((2, 1, 3))(h_layer)
        # flatten the variates across all nodes, shape B x T' x (N V)
        h_layer = Reshape((-1, self.n_nodes * self.variates))(h_layer)
        print(h_layer.shape)
        for layer in self._lstm_layers:
            h_layer = layer(h_layer)

        # h_layer = Dropout(self.dropout)(h_layer)
        h_layer = BatchNormalization()(h_layer)
        h_layer = self._decoder_layer(h_layer)
        h_layer = Reshape((self.n_nodes, self.forecast_horizon))(h_layer)

        if self.multivariate_input:
            # flatten things out to the multivariate shape
            # shape B x N x V
            h_layer = Reshape((self.n_nodes, self.variates))(h_layer)

        return h_layer

    def in_out_tensors(self):
        """
        Builds a GCN model for node  feature prediction
        Returns:
            tuple: ``(x_inp, x_out)``, where ``x_inp`` is a list of Keras/TensorFlow
                input tensors for the GCN model and ``x_out`` is a tensor of the GCN model output.
        """
        # Inputs for features
        if self.multivariate_input:
            shape = (None, self.n_nodes, self.n_features, self.variates)
        else:
            shape = (None, self.n_nodes, self.n_features)

        x_t = Input(batch_shape=shape)

        # Indices to gather for model output
        out_indices_t = Input(batch_shape=(None, self.n_nodes), dtype="int32")

        x_inp = [x_t, out_indices_t]
        x_out = self(x_inp)

        return x_inp[0], x_out



def get_gcnlstm(forecast_horizon, num_lags, sensor_adj):
    gcn_lstm = GCN_LSTM(
        seq_len=num_lags,
        adj=sensor_adj,
        forecast_horizon=forecast_horizon,
        gc_layer_sizes=[64, 64],
        gc_activations=["relu", 'relu'],
        lstm_layer_sizes=[120],
        lstm_activations=["tanh"]
    )

    x_input, x_output = gcn_lstm.in_out_tensors()
    model = Model(inputs=x_input, outputs=x_output)
    return model
