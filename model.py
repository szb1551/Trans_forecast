import numpy as np
import torch.nn as nn
import torch, math
import time
import gcn
import causal_convolution_layer
from models.ConvLstm import ConvLSTM

# from torchkeras import summary

"""
The architecture is based on the paper “Attention Is All You Need”. 
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2

        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps
        reshaped_input = input_seq.contiguous().view(-1, input_seq.shape[2], input_seq.shape[3], input_seq.shape[4])

        output = self.module(reshaped_input)
        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            output = output.contiguous().view(input_seq.size(0), -1, output.shape[1], output.shape[2], output.shape[3])
        else:
            # (timesteps, samples, output_size)
            output = output.contiguous().view(-1, input_seq.size(1), output.size(-1))
        return output


class TimeDistributed_FL(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed_FL, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2

        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps
        reshaped_input = input_seq.contiguous().view(-1, input_seq.shape[2], input_seq.shape[3], input_seq.shape[4])

        output = self.module(reshaped_input)
        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            output = output.contiguous().view(input_seq.size(0), -1, output.size(-1))
        else:
            # (timesteps, samples, output_size)
            output = output.contiguous().view(-1, input_seq.size(1), output.size(-1))
        return output


class TimeDistributed_Batch(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed_Batch, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, input_seq):
        assert len(input_seq.shape) > 2

        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps
        reshaped_input = input_seq.contiguous().view(-1, input_seq.shape[-3], input_seq.shape[-2], input_seq.shape[-1])

        output = self.module(reshaped_input)
        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            output = output.contiguous().view(input_seq.shape[0], -1, output.shape[-3], output.shape[-2],
                                              output.shape[-1])
        else:
            # (timesteps, samples, output_size)
            output = output.contiguous().view(-1, input_seq.shape[0], output.shape[-3], output.shape[-2],
                                              output.shape[-1])
        return output


class baseline_conv(nn.Module):
    def __init__(self, args):
        super(baseline_conv, self).__init__()

        self.Conv = TimeDistributed(nn.Conv2d(args['conv_feature'], args['filters'],
                                              args['conv_kernel'], padding=args['padding']))
        self.flatten = TimeDistributed_FL(nn.Flatten())
        self.batch_norm = nn.BatchNorm1d(args['train_length'])
        self.LSTM = nn.LSTM(input_size=args['filters'] * args['input_shape'][-2] * args['input_shape'][-1],
                            hidden_size=args['lstm_size'], num_layers=args['num_lstm'], dropout=args['dropout'])
        self.batch_norm2 = nn.BatchNorm1d(args['lstm_size'])
        self.fc1 = nn.Linear(args['lstm_size'],
                             args['forcast_window'] * args['input_shape'][-2] * args['input_shape'][-1])

    def forward(self, x=torch.zeros((3, 30, 5, 5))):  # x [B, T, 5, 5]
        x = self.Conv(x.unsqueeze(2))  # x[B, T, 1, 5, 5]->[B,T,16,5,5]
        x = self.flatten(x)  # [B,T,400]
        x = self.batch_norm(x).permute(1, 0, 2)  # ->[T,B,400]
        x, (_, _) = self.LSTM(x)  # [T,B,hidden_size*num_directions]
        x = self.batch_norm2(x[-1])  # output[-1]才是lstm真正的隐层输出ht
        x = self.fc1(x)  # [B, 100]->[B, T*5*5]
        x = x.reshape(x.shape[0], -1, 5, 5)  # [B,T,5,5]
        return x


class baseline_conv2(nn.Module):
    def __init__(self, args):
        super(baseline_conv2, self).__init__()

        self.Conv = ConvLSTM(input_dim=args['input_dim'], hidden_dim=args['filters'], kernel_size=args['conv_kernel'],
                             num_layers=args['num_in_out_lstm'], batch_first=True)
        self.batch_norm = TimeDistributed_Batch(nn.BatchNorm2d(args['filters']))
        self.Conv2 = ConvLSTM(input_dim=args['filters'], hidden_dim=args['filters'], kernel_size=args['conv_kernel'],
                              num_layers=args['num_lstm'], batch_first=True)
        self.batch_norm2 = TimeDistributed_Batch(nn.BatchNorm2d(args['filters']))
        self.Conv3 = ConvLSTM(input_dim=args['filters'], hidden_dim=args['forcast_window'],
                              kernel_size=args['conv_kernel'], num_layers=args['num_in_out_lstm'], batch_first=True)

    def forward(self, x=torch.zeros((3, 30, 5, 5))):  # x [B, T, 5, 5]
        x, _ = self.Conv(x.unsqueeze(2))  # x[B, T, 1, 5, 5]->[B,T,16,5,5]
        x = self.batch_norm(x[-1])  # [B,T,16,5,5]
        x, _ = self.Conv2(x)  # ->[B,T,16,5,5]
        x = self.batch_norm2(x[-1])  # [B,T,16,5,5]
        _, x = self.Conv3(x)  # [B,T,7,5,5]
        return x[-1][-1]


class baseline_gcn(nn.Module):
    def __init__(self, args):
        super(baseline_gcn, self).__init__()
        self.gcn_enc = gcn.GCN_BASELINE(args['adj_matrix'], [args['enc_filters'], args['train_length']],
                                        [args['train_length'], args['enc_filters']], [nn.ReLU(), nn.ReLU()])
        self.LSTM = nn.LSTM(args['adj_matrix'].shape[0], args['lstm_size'], args['num_lstm'], dropout=args['dropout'])
        self.fc1 = nn.Linear(args['lstm_size'], args['adj_matrix'].shape[0] * args['forcast_window'])

    def forward(self, x=torch.zeros((3, 47, 30))):
        x = self.gcn_enc(x)  # [B, N, F] -> [B, N, F]
        x, (_, _) = self.LSTM(x.permute(2, 0, 1))  # [F, B, N] output(x, (h0, c0))
        x = self.fc1(x[-1])  # [B,N]->N[B, T*N]
        x = x.reshape(x.shape[0], 47, -1)
        return x


class Transformer(torch.nn.Module):
    # d_model : number of features
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.input_embedding = causal_convolution_layer.context_embedding(args['conv_feature'],
                                                                          args['filters'], args['conv_kernel'])
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=args['filters'],
                                                              nhead=args['n_head'], dropout=args['dropout'])
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=args['num_layers'])
        self.positional_embedding = torch.nn.Embedding(args['embedding_size'], args['filters'])
        # self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=1, dropout=dropout)
        # self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.decoder_layer = torch.nn.TransformerDecoderLayer(d_model=args['filters'], nhead=args['n_head'],
                                                              dropout=args['dropout'])
        self.transformer_decoder = torch.nn.TransformerDecoder(self.decoder_layer, num_layers=args['num_layers'])
        self.fc1 = torch.nn.Linear(args['filters'], 1)
        self.train_length = args['train_length']
        self.forcast_window = args['forcast_window']
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        # torch.triu() 返回右上三角
        mask = (torch.triu(torch.ones(sz, sz) == 1)).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _generate_forward_mask(self, sz):
        mask = torch.zeros(sz, sz)
        for i in range(self.train_length):
            mask[i][self.train_length:] = 1
        for i in range(self.train_length, sz):
            mask[i][i + 1:] = 1
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x, src, tar):
        mask = self._generate_square_subsequent_mask(tar.shape[1]).to(tar.device)
        # print(src.shape) #[batch, time_step]
        # print(x.shape)
        x_enc = x[:, :self.train_length]  # [batch, 30]
        x_dec = x[:, self.train_length - 1:self.train_length - 1 + tar.shape[1]]  # [batch, 7]
        z_enc = torch.cat((src.unsqueeze(1), x_enc.unsqueeze(1)), dim=1)  # [batch, feature, time_step]
        z_enc_embedding = self.input_embedding(z_enc).permute(2, 0, 1)  # [time_step, batch, feature]

        # [batch, time, feature] -> [time, batch, feature]
        enc_positional_embeddings = self.positional_embedding(x_enc.type(torch.long)).permute(1, 0, 2)

        z_dec = torch.cat((tar.unsqueeze(1), x_dec.unsqueeze(1)), dim=1)
        z_dec_embedding = self.input_embedding(z_dec).permute(2, 0, 1)
        dec_positional_embeddings = self.positional_embedding(x_dec.type(torch.long)).permute(1, 0, 2)
        # print(x_dec.type(torch.long))
        input_embedding = z_enc_embedding + enc_positional_embeddings
        tar_embedding = z_dec_embedding + dec_positional_embeddings
        enc_output = self.transformer_encoder(input_embedding)
        output = self.transformer_decoder(tgt=tar_embedding, memory=enc_output, tgt_mask=mask)
        # output = self.decoder(tgt=src, memory=output)
        output = self.fc1(output.permute(1, 0, 2))  # 【batch, time, 1】
        return output


class Transformer2(torch.nn.Module):  # 加入cnn卷积网络
    # d_model : number of features
    def __init__(self, args):
        super(Transformer2, self).__init__()
        self.input_embedding = TimeDistributed(nn.Conv2d(args['conv_feature'], args['filters'], args['conv_kernel'],
                                                         padding=args['padding']))
        self.flatten = TimeDistributed_FL(nn.Flatten())
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=args['filters'] * args['input_shape'][-2] * args['input_shape'][-1],
            nhead=args['n_head'], dropout=args['dropout'])
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=args['num_layers'])
        self.positional_embedding = torch.nn.Embedding(
            args['embedding_size'], args['filters'] * args['input_shape'][-2] * args['input_shape'][-1])
        # self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=1, dropout=dropout)
        # self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=args['filters'] * args['input_shape'][-2] * args['input_shape'][-1],
            nhead=args['n_head'], dropout=args['dropout'])
        self.transformer_decoder = torch.nn.TransformerDecoder(self.decoder_layer, num_layers=args['num_layers'])
        self.fc1 = torch.nn.Linear(args['filters'] * args['input_shape'][-2] * args['input_shape'][-1],
                                   args['input_shape'][-2] * args['input_shape'][-1])
        self.train_length = args['train_length']
        self.forcast_window = args['forcast_window']
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        # torch.triu() 返回右上三角
        mask = (torch.triu(torch.ones(sz, sz) == 1)).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, src=torch.zeros((3, 30, 5, 5)), tar=torch.zeros((3, 7, 5, 5))):
        mask = self._generate_square_subsequent_mask(tar.shape[1]).to(tar.device)
        # print(src.shape) #[batch, time_step, 5, 5]
        # print(x.shape)
        x_enc = x[:, :self.train_length]  # [batch, 30]
        x_dec = x[:, self.train_length - 1:self.train_length - 1 + tar.shape[1]]  # [batch, 7]
        z_enc = self.input_embedding(src.unsqueeze(2))  # [batch, timestep, feature_conv,5,5]
        z_enc_embedding = self.flatten(z_enc).permute(1, 0, 2)  # [timestep, batch, feature]
        # [batch, time, feature] -> [time, batch, feature]
        enc_positional_embeddings = self.positional_embedding(x_enc.type(torch.long)).permute(1, 0, 2)

        z_dec = self.input_embedding(tar.unsqueeze(2))
        z_dec_embedding = self.flatten(z_dec).permute(1, 0, 2)
        dec_positional_embeddings = self.positional_embedding(x_dec.type(torch.long)).permute(1, 0, 2)
        # print(x_dec.type(torch.long))
        input_embedding = z_enc_embedding + enc_positional_embeddings
        tar_embedding = z_dec_embedding + dec_positional_embeddings
        enc_output = self.transformer_encoder(input_embedding)
        output = self.transformer_decoder(tgt=tar_embedding, memory=enc_output, tgt_mask=mask)
        # output = self.decoder(tgt=src, memory=output)
        output = self.fc1(output.permute(1, 0, 2))  # 【batch, time, 5*5】
        output = output.reshape((output.shape[0], -1, 5, 5))  # [batch, time, 5, 5]
        return output


class Transformer3(torch.nn.Module):  # 加入gcn卷积网络
    # d_model : number of features
    # fiture维度不能变动，想加入transformer的话，即时间维度不能当作特征维度
    def __init__(self, args):
        super(Transformer3, self).__init__()
        # B, N, T, F
        # self.input_embedding = TimeDistributed(nn.Conv2d(1, 16, 3, padding=1))
        # self.flatten = TimeDistributed_FL(nn.Flatten())
        self.gcn_enc = gcn.GCN_FIGURE(args['adj_matrix'], [args['enc_filters'], args['train_length']],
                                      [args['conv_feature'], args['enc_filters']], [nn.ReLU(), nn.ReLU()])
        self.gcn_dec = gcn.GCN_FIGURE(args['adj_matrix'], [args['dec_filters'], args['forcast_window']],
                                      [args['conv_feature'], args['dec_filters']], [nn.ReLU(), nn.ReLU()])
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=args['embedding_feature'],
                                                              nhead=args['n_head'], dropout=args['dropout'])
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=args['num_layers'])
        self.positional_embedding = torch.nn.Embedding(args['embedding_size'], args['embedding_feature'])
        self.decoder_layer = torch.nn.TransformerDecoderLayer(d_model=args['embedding_feature'], nhead=args['n_head'],
                                                              dropout=args['dropout'])
        self.transformer_decoder = torch.nn.TransformerDecoder(self.decoder_layer, num_layers=args['num_layers'])
        self.fc_enc_gcn = torch.nn.Linear(args['adj_matrix'].shape[0] * args['train_length'],
                                          args['embedding_feature'])
        self.fc_dec_gcn = torch.nn.Linear(args['adj_matrix'].shape[0] * args['forcast_window'],
                                          args['embedding_feature'])
        self.fc1 = torch.nn.Linear(args['embedding_feature'], args['adj_matrix'].shape[0])
        self.train_length = args['train_length']
        self.forcast_window = args['forcast_window']
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        # torch.triu() 返回右上三角
        mask = (torch.triu(torch.ones(sz, sz) == 1)).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _generate_forward_mask(self, sz):
        mask = torch.zeros(sz, sz)
        for i in range(self.train_length):
            mask[i][self.train_length:] = 1
        for i in range(self.train_length, sz):
            mask[i][i + 1:] = 1
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x, src=torch.zeros((3, 47, 30)), tar=torch.zeros((3, 47, 7))):
        mask = self._generate_square_subsequent_mask(tar.shape[2]).to(tar.device)
        # print(src.shape) #[batch, 47, 30]
        x_enc = x[:, :self.train_length]  # [batch, 30]
        x_dec = x[:, self.train_length - 1:self.train_length - 1 + tar.shape[2]]  # [batch, 7]
        z_enc = self.gcn_enc(src.unsqueeze(-1))  # [batch, N, T, F]
        z_enc = z_enc.permute(2, 0, 1, 3).reshape(z_enc.shape[2], z_enc.shape[0], -1)  # [T,B,N,F]->[T,B,NF]
        z_enc_embedding = self.fc_enc_gcn(z_enc)  # [time, batch, 128]
        # [batch, time, feature] -> [time, batch, feature]
        enc_positional_embeddings = self.positional_embedding(x_enc.type(torch.long)).permute(1, 0, 2)

        z_dec = self.gcn_dec(tar.unsqueeze(-1))
        z_dec = z_dec.permute(2, 0, 1, 3).reshape(z_dec.shape[2], z_dec.shape[0], -1)  # [T,B,N,F]->[T,B,NF]
        z_dec_embedding = self.fc_dec_gcn(z_dec)
        dec_positional_embeddings = self.positional_embedding(x_dec.type(torch.long)).permute(1, 0, 2)
        input_embedding = z_enc_embedding + enc_positional_embeddings
        tar_embedding = z_dec_embedding + dec_positional_embeddings
        enc_output = self.transformer_encoder(input_embedding)
        output = self.transformer_decoder(tgt=tar_embedding, memory=enc_output, tgt_mask=mask)  # [time, batch, 128]
        # output = self.decoder(tgt=src, memory=output)
        output = self.fc1(output).permute(1, 2, 0)  # 【batch, 47, time】
        return output


class Transformer3_add(torch.nn.Module):  # 加入gcn卷积网络,与汽车保有量
    # d_model : number of features
    # fiture维度不能变动，想加入transformer的话，即时间维度不能当作特征维度
    def __init__(self, args):
        super(Transformer3_add, self).__init__()
        # B, N, T, F
        # self.input_embedding = TimeDistributed(nn.Conv2d(1, 16, 3, padding=1))
        # self.flatten = TimeDistributed_FL(nn.Flatten())
        self.gcn_enc = gcn.GCN_FIGURE(args['adj_matrix'], [args['enc_filters'], args['train_length']],
                                      [args['conv_feature'], args['enc_filters']], [nn.ReLU(), nn.ReLU()])
        self.gcn_dec = gcn.GCN_FIGURE(args['adj_matrix'], [args['dec_filters'], args['forcast_window']],
                                      [args['conv_feature'], args['dec_filters']], [nn.ReLU(), nn.ReLU()])
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=args['embedding_feature'],
                                                              nhead=args['n_head'], dropout=args['dropout'])
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=args['num_layers'])
        self.positional_embedding = torch.nn.Embedding(args['embedding_size'], args['embedding_feature'])
        self.decoder_layer = torch.nn.TransformerDecoderLayer(d_model=args['embedding_feature'], nhead=args['n_head'],
                                                              dropout=args['dropout'])
        self.transformer_decoder = torch.nn.TransformerDecoder(self.decoder_layer, num_layers=args['num_layers'])
        self.fc_enc_gcn = torch.nn.Linear(args['adj_matrix'].shape[0] * (args['train_length'] + args['other']),
                                          args['embedding_feature'])
        self.fc_dec_gcn = torch.nn.Linear(args['adj_matrix'].shape[0] * args['forcast_window'],
                                          args['embedding_feature'])
        self.fc1 = torch.nn.Linear(args['embedding_feature'], args['adj_matrix'].shape[0])
        self.train_length = args['train_length']
        self.forcast_window = args['forcast_window']
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        # torch.triu() 返回右上三角
        mask = (torch.triu(torch.ones(sz, sz) == 1)).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _generate_forward_mask(self, sz):
        mask = torch.zeros(sz, sz)
        for i in range(self.train_length):
            mask[i][self.train_length:] = 1
        for i in range(self.train_length, sz):
            mask[i][i + 1:] = 1
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x, y=torch.zeros(3, 30, 1), src=torch.zeros((3, 47, 30)), tar=torch.zeros((3, 47, 7))):
        mask = self._generate_square_subsequent_mask(tar.shape[2]).to(tar.device)
        # print(src.shape) #[batch, 47, 30]
        x_enc = x[:, :self.train_length]  # [batch, 30]
        x_dec = x[:, self.train_length - 1:self.train_length - 1 + tar.shape[2]]  # [batch, 7]
        z_enc = self.gcn_enc(src.unsqueeze(-1))  # [batch, N, T, F]
        B, N, T = z_enc.size(0), z_enc.size(1), z_enc.size(2)
        z_enc_time = y.unsqueeze(1).expand(B, N, T, 1)  # [batch, :, T ,1]; # 添加的时间汽车保有量维度
        z_enc = torch.cat((z_enc, z_enc_time), -1)  # [batch, N, T, F+1];
        z_enc = z_enc.permute(2, 0, 1, 3).reshape(z_enc.shape[2], z_enc.shape[0], -1)  # [T,B,N,F]->[T,B,NF]
        z_enc_embedding = self.fc_enc_gcn(z_enc)  # [time, batch, 128]
        # [batch, time, feature] -> [time, batch, feature]
        enc_positional_embeddings = self.positional_embedding(x_enc.type(torch.long)).permute(1, 0, 2)

        z_dec = self.gcn_dec(tar.unsqueeze(-1))
        z_dec = z_dec.permute(2, 0, 1, 3).reshape(z_dec.shape[2], z_dec.shape[0], -1)  # [T,B,N,F]->[T,B,NF]
        z_dec_embedding = self.fc_dec_gcn(z_dec)
        dec_positional_embeddings = self.positional_embedding(x_dec.type(torch.long)).permute(1, 0, 2)
        input_embedding = z_enc_embedding + enc_positional_embeddings
        tar_embedding = z_dec_embedding + dec_positional_embeddings
        enc_output = self.transformer_encoder(input_embedding)
        output = self.transformer_decoder(tgt=tar_embedding, memory=enc_output, tgt_mask=mask)  # [time, batch, 128]
        # output = self.decoder(tgt=src, memory=output)
        output = self.fc1(output).permute(1, 2, 0)  # 【batch, 47, time】
        return output


class Transformer3_diff(torch.nn.Module):  # 加入gcn卷积网络
    # d_model : number of features
    # fiture维度不能变动，想加入transformer的话，即时间维度不能当作特征维度
    def __init__(self, args):
        super(Transformer3_diff, self).__init__()
        # B, N, T, F
        # self.input_embedding = TimeDistributed(nn.Conv2d(1, 16, 3, padding=1))
        # self.flatten = TimeDistributed_FL(nn.Flatten())
        self.gcn_enc = gcn.GCN_FIGURE(args['adj_matrix'], [args['enc_filters'], args['train_length']],
                                      [args['conv_feature'], args['enc_filters']], [nn.ReLU(), nn.ReLU()])
        self.gcn_dec = gcn.GCN_FIGURE(args['adj_matrix'], [args['dec_filters'], args['forcast_window']],
                                      [args['conv_feature'], args['dec_filters']], [nn.ReLU(), nn.ReLU()])
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=args['embedding_feature'],
                                                              nhead=args['n_head'], dropout=args['dropout'])
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=args['num_layers'])
        self.positional_embedding = torch.nn.Embedding(args['embedding_size'], args['embedding_feature'])
        self.decoder_layer = torch.nn.TransformerDecoderLayer(d_model=args['embedding_feature'], nhead=args['n_head'],
                                                              dropout=args['dropout'])
        self.transformer_decoder = torch.nn.TransformerDecoder(self.decoder_layer, num_layers=args['num_layers'])
        self.fc_enc_gcn = torch.nn.Linear(args['adj_matrix'].shape[0] * (args['train_length']),
                                          args['embedding_feature'])
        self.fc_dec_gcn = torch.nn.Linear(args['adj_matrix'].shape[0] * (args['forcast_window']),
                                          args['embedding_feature'])
        self.fc1 = torch.nn.Linear(args['embedding_feature'], args['adj_matrix'].shape[0])
        self.train_length = args['train_length']
        self.forcast_window = args['forcast_window']
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        # torch.triu() 返回右上三角
        mask = (torch.triu(torch.ones(sz, sz) == 1)).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _generate_forward_mask(self, sz):
        mask = torch.zeros(sz, sz)
        for i in range(self.train_length):
            mask[i][self.train_length:] = 1
        for i in range(self.train_length, sz):
            mask[i][i + 1:] = 1
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x, src=torch.zeros((3, 47, 30)), tar=torch.zeros((3, 47, 7))):
        mask = self._generate_square_subsequent_mask(tar.shape[2]).to(tar.device)
        # print(src.shape) #[batch, 47, 30]
        x_enc = x[:, :self.train_length]  # [batch, 30]
        x_dec = x[:, self.train_length - 1:self.train_length - 1 + tar.shape[2]]  # [batch, 7]
        z_enc = self.gcn_enc(src.unsqueeze(-1))  # [batch, N, T, F]
        z_enc = z_enc.permute(2, 0, 1, 3).reshape(z_enc.shape[2], z_enc.shape[0], -1)  # [T,B,N,F]->[T,B,NF]
        z_enc_embedding = self.fc_enc_gcn(z_enc)  # [time, batch, 128]
        # [batch, time, feature] -> [time, batch, feature]
        enc_positional_embeddings = self.positional_embedding(x_enc.type(torch.long)).permute(1, 0, 2)

        z_dec = self.gcn_dec(tar.unsqueeze(-1))
        z_dec = z_dec.permute(2, 0, 1, 3).reshape(z_dec.shape[2], z_dec.shape[0], -1)  # [T,B,N,F]->[T,B,NF]
        z_dec_embedding = self.fc_dec_gcn(z_dec)
        dec_positional_embeddings = self.positional_embedding(x_dec.type(torch.long)).permute(1, 0, 2)
        input_embedding = z_enc_embedding + enc_positional_embeddings
        tar_embedding = z_dec_embedding + dec_positional_embeddings
        enc_output = self.transformer_encoder(input_embedding)
        output = self.transformer_decoder(tgt=tar_embedding, memory=enc_output, tgt_mask=mask)  # [time, batch, 128]
        # output = self.decoder(tgt=src, memory=output)
        output = self.fc1(output).permute(1, 2, 0)  # 【batch, 47, time】
        return output


class Transformer_Dalian(torch.nn.Module):
    # d_model : number of features
    # 用作专门处理大连电氢负荷数据的模型
    def __init__(self, args):
        super(Transformer_Dalian, self).__init__()
        # B, N, T, F
        # self.input_embedding = TimeDistributed(nn.Conv2d(1, 16, 3, padding=1))
        # self.flatten = TimeDistributed_FL(nn.Flatten())
        self.gcn_enc = gcn.GCN_FIGURE(args['adj_matrix'], [args['enc_filters'], args['enc_last']],
                                      [args['conv_feature'] + args['Hydrogen'],
                                       (args['conv_feature'] + args['Hydrogen']) * args['enc_filters']],
                                      [nn.ReLU(), nn.ReLU()], variates=1 + args['Hydrogen'])
        self.gcn_dec = gcn.GCN_FIGURE(args['adj_matrix'], [args['dec_filters'], args['forcast_window']],
                                      [args['conv_feature'] + args['Hydrogen'],
                                       (args['conv_feature'] + args['Hydrogen']) * args['dec_filters']],
                                      [nn.ReLU(), nn.ReLU()], variates=1 + args['Hydrogen'])
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=args['embedding_feature'],
                                                              nhead=args['n_head'], dropout=args['dropout'])
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=args['num_layers'])
        self.positional_embedding = torch.nn.Embedding(args['embedding_size'], args['embedding_feature'])
        self.decoder_layer = torch.nn.TransformerDecoderLayer(d_model=args['embedding_feature'], nhead=args['n_head'],
                                                              dropout=args['dropout'])
        self.transformer_decoder = torch.nn.TransformerDecoder(self.decoder_layer, num_layers=args['num_layers'])
        self.fc_enc_gcn = torch.nn.Linear(
            (args['conv_feature'] + args['Hydrogen']) * args['adj_matrix'].shape[0] * args['enc_last'],
            args['embedding_feature'])
        self.fc_dec_gcn = torch.nn.Linear(
            (args['conv_feature'] + args['Hydrogen']) * args['adj_matrix'].shape[0] * args['forcast_window'],
            args['embedding_feature'])
        self.fc1 = torch.nn.Linear(args['embedding_feature'],
                                   args['adj_matrix'].shape[0] * (args['conv_feature'] + args['Hydrogen']))
        self.train_length = args['train_length']
        self.forcast_window = args['forcast_window']
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        # torch.triu() 返回右上三角
        mask = (torch.triu(torch.ones(sz, sz) == 1)).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _generate_forward_mask(self, sz):
        mask = torch.zeros(sz, sz)
        for i in range(self.train_length):
            mask[i][self.train_length:] = 1
        for i in range(self.train_length, sz):
            mask[i][i + 1:] = 1
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x, src=torch.zeros((3, 47, 30, 2)), tar=torch.zeros((3, 47, 7, 2))):
        mask = self._generate_square_subsequent_mask(tar.shape[2]).to(tar.device)
        # print(src.shape) #[batch, Node, Time, feature]
        x_enc = x[:, :self.train_length]  # [batch, 30]
        if self.train_length - 1 + tar.shape[2] <= x.shape[-1]:
            x_dec = x[:, self.train_length - 1:self.train_length - 1 + tar.shape[2]]  # [batch, 7]
        else:
            x_dec = x[:, self.train_length - 1:self.train_length - 1 + tar.shape[2]]  # [batch, 7]
            extra_values = torch.zeros((x.shape[0], self.train_length - 1 + tar.shape[2] - x.shape[-1]))
            for i in range(self.train_length - 1 + tar.shape[2] - x.shape[-1]):
                extra_values[:, i] = x[:, -1] + i + 1
            x_dec = torch.cat((x_dec, extra_values.to(tar.device)), dim=-1)
        z_enc = self.gcn_enc(src)  # [batch, N, T, F]
        z_enc = z_enc.permute(2, 0, 1, 3).reshape(z_enc.shape[2], z_enc.shape[0], -1)  # [T,B,N,F]->[T,B,NF]
        z_enc_embedding = self.fc_enc_gcn(z_enc)  # [time, batch, 128]
        # [batch, time, feature] -> [time, batch, feature]
        enc_positional_embeddings = self.positional_embedding(x_enc.type(torch.long)).permute(1, 0, 2)

        z_dec = self.gcn_dec(tar)
        z_dec = z_dec.permute(2, 0, 1, 3).reshape(z_dec.shape[2], z_dec.shape[0], -1)  # [T,B,N,F]->[T,B,NF]
        z_dec_embedding = self.fc_dec_gcn(z_dec)
        dec_positional_embeddings = self.positional_embedding(x_dec.type(torch.long)).permute(1, 0, 2)
        input_embedding = z_enc_embedding + enc_positional_embeddings
        tar_embedding = z_dec_embedding + dec_positional_embeddings
        enc_output = self.transformer_encoder(input_embedding)
        output = self.transformer_decoder(tgt=tar_embedding, memory=enc_output, tgt_mask=mask)  # [time, batch, 128]
        output = self.fc1(output)  # 【time, batch, node*feature】
        output = output.reshape((output.shape[0], output.shape[1], -1, 2))  # [time, batch, nodes, 2]
        output = output.permute(1, 2, 0, 3)  # [batch, nodes, time , other]
        return output


class Transformer4(torch.nn.Module):
    # d_model : number of features
    def __init__(self, args):
        super(Transformer4, self).__init__()
        self.input_embedding = causal_convolution_layer.context_embedding(args['conv_feature'],
                                                                          args['filters'], args['conv_kernel'])
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=args['filters'],
                                                              nhead=args['n_head'], dropout=args['dropout'])
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=args['num_layers'])
        self.positional_embedding = torch.nn.Embedding(args['embedding_size'], args['filters'])
        # self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=1, dropout=dropout)
        # self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.decoder_layer = torch.nn.TransformerDecoderLayer(d_model=args['filters'], nhead=args['n_head'],
                                                              dropout=args['dropout'])
        self.transformer_decoder = torch.nn.TransformerDecoder(self.decoder_layer, num_layers=args['num_layers'])
        self.fc1 = torch.nn.Linear(args['filters'], 47)
        self.train_length = args['train_length']
        self.forcast_window = args['forcast_window']
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc1.bias.data.zero_()
        self.fc1.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        # torch.triu() 返回右上三角
        mask = (torch.triu(torch.ones(sz, sz) == 1)).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _generate_forward_mask(self, sz):
        mask = torch.zeros(sz, sz)
        for i in range(self.train_length):
            mask[i][self.train_length:] = 1
        for i in range(self.train_length, sz):
            mask[i][i + 1:] = 1
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x, src, tar):
        mask = self._generate_square_subsequent_mask(tar.shape[2]).to(tar.device)
        # print(src.shape) #[batch, nodes, time_step]
        # print(x.shape)
        x_enc = x[:, :self.train_length]  # [batch, 30]
        x_dec = x[:, self.train_length - 1:self.train_length - 1 + tar.shape[2]]  # [batch, 7]
        z_enc = torch.cat((src, x_enc.unsqueeze(1)), dim=1)  # [batch, nodes+feature, time_step]
        z_enc_embedding = self.input_embedding(z_enc).permute(2, 0, 1)  # [time_step, batch, feature]

        # [batch, time, feature] -> [time, batch, feature]
        enc_positional_embeddings = self.positional_embedding(x_enc.type(torch.long)).permute(1, 0, 2)

        z_dec = torch.cat((tar, x_dec.unsqueeze(1)), dim=1)
        z_dec_embedding = self.input_embedding(z_dec).permute(2, 0, 1)
        dec_positional_embeddings = self.positional_embedding(x_dec.type(torch.long)).permute(1, 0, 2)
        # print(x_dec.type(torch.long))
        input_embedding = z_enc_embedding + enc_positional_embeddings
        tar_embedding = z_dec_embedding + dec_positional_embeddings
        enc_output = self.transformer_encoder(input_embedding)
        output = self.transformer_decoder(tgt=tar_embedding, memory=enc_output, tgt_mask=mask)
        # output = self.decoder(tgt=src, memory=output)
        output = self.fc1(output).permute(1, 2, 0)  # 【batch, 47, time】
        return output


if __name__ == '__main__':
    train_length = 30
    forcast_window = 7
    A = torch.zeros(47, 47)
    # summary(model, torch.zeros((3, 37)).to(device))
