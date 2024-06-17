from model import Transformer, Transformer2, Transformer3, Transformer4, Transformer3_diff, Transformer3_add, \
    Transformer_Dalian
from model import baseline_conv, baseline_gcn, baseline_conv2

"""
    获取模型参数，模型参数设置 
"""


def select_model(name, args):
    if name == 'conv_lstm':
        model = baseline_conv(args)
    elif name == 'conv_lstm2':
        model = baseline_conv2(args)
    elif name == 'gcn_lstm':
        model = baseline_gcn(args)
    elif name == 'Transformer':
        model = Transformer(args)
    elif name == 'Transformer_cnn':
        model = Transformer2(args)
    elif name == 'Transformer_gcn':
        model = Transformer3(args)
    elif name == 'Transformer_gcn_add':
        model = Transformer3_add(args)
    elif name == 'Transformer_gcn_diff':
        model = Transformer3_diff(args)
    elif name == 'Transformer_all':
        model = Transformer4(args)
    elif name == "Transformer_gcn_Dalian":
        model = Transformer_Dalian(args)
    else:
        raise ValueError("传入无效参数{}".format(name))
    return model


def get_model_args(name, train_length, forcast_window, X_train=None, adj_matrix=None):
    if name == 'conv_lstm':
        return get_conv_lstm_args(train_length, forcast_window, X_train)
    elif name == 'conv_lstm2':
        return get_conv_lstm_2args(train_length, forcast_window, X_train)
    elif name == 'gcn_lstm':
        if adj_matrix is not None:
            return get_gcn_lstm_args(train_length, forcast_window, X_train, adj_matrix)
        else:
            raise ValueError("{}缺少邻接矩阵A".format('name'))
    elif name == 'Transformer':
        return get_Transformer_args(train_length, forcast_window, X_train)
    elif name == "Transformer_all":
        return get_Transformer_all_args(train_length, forcast_window, X_train)
    elif name == 'Transformer_cnn':
        return get_Transformer_cnn_args(train_length, forcast_window, X_train)
    elif name == 'Transformer_gcn':
        if adj_matrix is not None:
            return get_Transformer_gcn_args(train_length, forcast_window, X_train, adj_matrix)
        else:
            raise ValueError("{}缺少邻接矩阵A".format('name'))
    elif name == 'Transformer_gcn_add':
        if adj_matrix is not None:
            return get_Transformer_gcn_args(train_length, forcast_window, X_train, adj_matrix)
        else:
            raise ValueError("{}缺少邻接矩阵A".format('name'))
    elif name == "Transformer_gcn_Dalian":
        if adj_matrix is not None:
            return get_Transformer_Dalian_gcn_args(train_length, forcast_window, X_train, adj_matrix)
        else:
            raise ValueError("{}缺少邻接矩阵A".format('name'))
    elif name == 'Transformer_gcn_diff':
        if adj_matrix is not None:
            return get_Transformer_gcn_diff_args(train_length, forcast_window, X_train, adj_matrix)
        else:
            raise ValueError("{}缺少邻接矩阵A".format('name'))
    return


def get_conv_lstm_args(train_length, forcast_window, X_train):
    args = {'filters': 16, "train_length": train_length, "forcast_window": forcast_window,
            "input_shape": X_train.shape, 'lr': 1e-5, 'epochs': 10000, "lstm_size": 100,
            'conv_feature': 1, 'conv_kernel': 3, 'dropout': 0.1, 'num_lstm': 10, 'padding': 1,
            'name': 'conv_lstm', 'weight_decay': 1e-8}
    return args


def get_conv_lstm_2args(train_length, forcast_window, X_train):
    args = {'filters': 16, "train_length": train_length, "forcast_window": forcast_window,
            "input_shape": X_train.shape, 'lr': 1e-5, 'epochs': 10000, "input_dim": 1,
            'conv_kernel': (3, 3), 'num_lstm': 2, 'num_in_out_lstm': 1,
            'name': 'conv_lstm2', 'weight_decay': 1e-8}
    return args


def get_gcn_lstm_args(train_length, forcast_window, X_train, adj_matrix):
    args = {'train_length': train_length, 'forcast_window': forcast_window, 'input_shape': X_train.shape,
            'lr': 1e-4, 'epochs': 10000, 'lstm_size': 100, 'num_lstm': 10, 'dropout': 0.1, 'adj_matrix': adj_matrix,
            'enc_filters': 64, 'name': 'gcn_lstm', 'weight_decay': 1e-8}
    return args


def get_Transformer_args(train_length, forcast_window, X_train):
    args = {"train_length": train_length, "forcast_window": forcast_window, 'filters': 256,
            "input_shape": X_train.shape, 'lr': 1e-5, 'epochs': 10000, 'conv_feature': 2,
            'conv_kernel': 8, 'n_head': 8, 'num_layers': 4, 'dropout': 0.1, 'embedding_size': 200,
            'name': 'day', 'weight_decay': 1e-8}
    return args


def get_Transformer_all_args(train_length, forcast_window, X_train):
    args = {"train_length": train_length, "forcast_window": forcast_window, 'filters': 256,
            "input_shape": X_train.shape, 'lr': 1e-6, 'epochs': 1000, 'conv_feature': 48,
            'conv_kernel': 10, 'n_head': 8, 'num_layers': 4, 'dropout': 0.1, 'embedding_size': 200,
            'name': 'day_station', 'weight_decay': 1e-10}
    return args


def get_Transformer_cnn_args(train_length, forcast_window, X_train):
    args = {"train_length": train_length, "forcast_window": forcast_window, 'filters': 8,
            "input_shape": X_train.shape, 'lr': 1e-5, 'epochs': 1000, 'conv_feature': 1,
            'conv_kernel': 3, 'n_head': 8, 'num_layers': 4, 'dropout': 0.1, 'embedding_size': 200,
            'padding': 1, 'name': 'raster', 'weight_decay': 1e-9}
    return args


def get_Transformer_gcn_args(train_length, forcast_window, X_train, adj_matrix):
    args = {"train_length": train_length, "forcast_window": forcast_window, 'enc_filters': 64,
            "input_shape": X_train.shape, 'lr': 1e-4, 'epochs': 10000, 'conv_feature': 1,
            'n_head': 8, 'num_layers': 4, 'dropout': 0.1, 'dec_filters': 32,
            'adj_matrix': adj_matrix, 'embedding_size': 200, 'embedding_feature': 128,
            'name': 'gcn', 'weight_decay': 1e-9, 'other': 1}
    return args


def get_Transformer_Dalian_gcn_args(train_length, forcast_window, X_train, adj_matrix):
    args = {"train_length": train_length, "forcast_window": forcast_window, 'enc_filters': 64,
            "input_shape": X_train.shape, 'lr': 1e-4, 'epochs': 10, 'conv_feature': 1,
            'n_head': 8, 'num_layers': 4, 'dropout': 0.1, 'dec_filters': 32,
            'adj_matrix': adj_matrix, 'embedding_size': 200, 'embedding_feature': 512,
            'name': 'gcn', 'weight_decay': 1e-9, 'other': 1, 'Hydrogen': 1}
    return args


def get_Transformer_gcn_diff_args(train_length, forcast_window, X_train, adj_matrix):
    args = {"train_length": train_length - 1, "forcast_window": forcast_window - 1, 'enc_filters': 64,
            "input_shape": X_train.shape, 'lr': 1e-5, 'epochs': 10000, 'conv_feature': 1,
            'n_head': 8, 'num_layers': 4, 'dropout': 0.15, 'dec_filters': 32,
            'adj_matrix': adj_matrix, 'embedding_size': 200, 'embedding_feature': 128,
            'name': 'gcn_diff', 'weight_decay': 1e-9}
    return args


def get_Compare_model_args():
    args = {
        "model_initial_sequential": ["conv_lstm_path", "gcn_lstm_path", "day_path", "raster_day_path", "gcn_day_path"],
        "model_args_name": ["conv_lstm", "gcn_lstm", "Transformer", "Transformer_cnn", "Transformer_gcn"],
        "conv_lstm_path": "results/split_results/cnn_lstm/30+7_10000次/model/model_last_length30.h5",
        "gcn_lstm_path": "results/split_results/gcn_lstm/30+7_10000次/model/model_last_length30.h5",
        "day_path": "results/split_results/Trans/30+7_10000次/model/model_last_length30.h5",
        "raster_day_path": "results/split_results/cnn_Trans/30+7_10000次/model/model_last_length30.h5",
        "gcn_day_path": "results/split_results/gcn_Trans/30+7_10000次/model2/model_last_length30.h5",
        'train_length': 30, "forcast_window": 7, 'save_figure': False,
        "day_test_dataset": "split_data/Transformer_test_dataset.npy",
        "day_max_": "train_data/max_.npy",
        "raster_test_dataset": "split_data/Transformer_cnn_test_dataset.npy",
        "raster_max_": "train_data/max_raster.npy",
        "Graph": "train_data/GCN_Graph.gpickle",
        "gcn_test_dataset": "split_data/Transformer_gcn_test_dataset.npy",
        "gcn_max_": "train_data/G_max_.npy",
        "test_time_list": "split_data/Transformer_cnn_test_timeset.npy"
    }
    return args


def get_Compare_long_model_args():
    args = {
        "model_initial_sequential": ["conv_lstm_path", "gcn_lstm_path", "day_path", "raster_day_path", "gcn_day_path"],
        "model_args_name": ["conv_lstm", "gcn_lstm", "Transformer", "Transformer_cnn", "Transformer_gcn"],
        "conv_lstm_path": "results/split_results/cnn_lstm/120+30_10000次/model/model_last_length120.h5",
        "gcn_lstm_path": "results/split_results/gcn_lstm/120+30_10000次/model/model_last_length120.h5",
        "day_path": "results/split_results/Trans/120+30_10000次/model/model_last_length120.h5",
        "raster_day_path": "results/split_results/cnn_Trans/120+30_10000次/model/model_last_length120.h5",
        "gcn_day_path": "results/split_results/gcn_Trans/120+30_10000次/model/model_last_length120.h5",
        'train_length': 120, "forcast_window": 30, 'save_figure': False,
        "day_test_dataset": "split_data/Transformer_long_test_dataset.npy",
        "day_max_": "train_data/max_.npy",
        "raster_test_dataset": "split_data/Transformer_cnn_long_test_dataset.npy",
        "raster_max_": "train_data/max_raster.npy",
        "Graph": "train_data/GCN_Graph.gpickle",
        "gcn_test_dataset": "split_data/Transformer_gcn_long_test_dataset.npy",
        "gcn_max_": "train_data/G_max_.npy",
        "test_time_list": "split_data/Transformer_long_test_timeset.npy"
    }
    return args
