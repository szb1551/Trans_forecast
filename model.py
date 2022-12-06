import torch.nn as nn
import torch, math
import time
import causal_convolution_layer
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

class Transformer(torch.nn.Module):
    # d_model : number of features
    def __init__(self, train_length, forcast_window):
        super(Transformer, self).__init__()
        self.input_embedding = causal_convolution_layer.context_embedding(2, 256, 8)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=256, nhead=8, dropout=0.1)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.positional_embedding = torch.nn.Embedding(100, 256)
        # self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=1, dropout=dropout)
        # self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.decoder_layer = torch.nn.TransformerDecoderLayer(d_model=256, nhead=8, dropout=0.1)
        self.transformer_decoder = torch.nn.TransformerDecoder(self.decoder_layer, num_layers=4)
        self.fc1 = torch.nn.Linear(256, 1)
        self.train_length = train_length
        self.forcast_window = forcast_window
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
        mask = self._generate_square_subsequent_mask(tar.shape[1]).to(device)
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
    def __init__(self, train_length, forcast_window):
        super(Transformer2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(1, 16, 3, padding=1)
        self.input_embedding = TimeDistributed(nn.Conv2d(1, 16, 3, padding=1))
        self.flatten = TimeDistributed_FL(nn.Flatten())
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=400, nhead=8, dropout=0.1)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.positional_embedding = torch.nn.Embedding(100, 400)
        # self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=1, dropout=dropout)
        # self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.decoder_layer = torch.nn.TransformerDecoderLayer(d_model=400, nhead=8, dropout=0.1)
        self.transformer_decoder = torch.nn.TransformerDecoder(self.decoder_layer, num_layers=4)
        self.fc1 = torch.nn.Linear(400, 5 * 5)
        self.train_length = train_length
        self.forcast_window = forcast_window
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

    def forward(self, x, src=torch.zeros((3, 30, 5, 5)).to(device), tar=torch.zeros((3, 7, 5, 5)).to(device)):
        mask = self._generate_square_subsequent_mask(tar.shape[1]).to(device)
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


if __name__ == '__main__':
    train_length = 30
    forcast_window = 7
    model = Transformer2(30, 7).to(device)
    # print(model)
    # summary(model, torch.zeros((3, 37)).to(device))
