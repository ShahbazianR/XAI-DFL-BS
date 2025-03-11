import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from pytorch_tcn import TCN
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.feature_size = hidden_size

        # Linear transformations for Q, K, V from the same source
        self.key = nn.Linear(self.feature_size, self.feature_size)
        self.query = nn.Linear(self.feature_size, self.feature_size)
        self.value = nn.Linear(self.feature_size, self.feature_size)

    def forward(self, x, explain=False):
        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        # Multiply weights with values
        output = torch.matmul(attention_weights, values)
        
        if explain:
            return output, attention_weights
        else:
            return output


class BiLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, tcn_input_size, output_size, lr=0.001):
        super(BiLSTMAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr

        self.do = nn.Dropout1d()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=0.2)
        self.relu = nn.ReLU()
        self.bn_lstm = nn.BatchNorm1d(num_features=tcn_input_size)

        self.tcn = TCN(num_inputs=tcn_input_size, num_channels=[1, 2, 4], dilations=[1, 2, 4], activation='relu')
        self.tcn = TCN(num_inputs=tcn_input_size, num_channels=[1, 2], dilations=[1, 2], activation='relu')

        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.avgpool = nn.AvgPool1d(kernel_size=4, stride=4)

        self.bn_tcn= nn.BatchNorm1d(num_features=2)

        self.flatten = nn.Flatten()

        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.sigmoid = nn.Sigmoid()

        self.loss_fn = nn.MSELoss()  # binary cross entropy
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x, explain=False):
        # x = self.do(x)
        lstm_out = self.relu(self.bn_lstm(self.lstm1(x)[0]))
        lstm_out = self.relu(self.bn_lstm(self.lstm2(lstm_out)[0]))
        
        tcn_output = self.bn_tcn(self.tcn(lstm_out))

        maxpool = self.maxpool(tcn_output)
        avgpool = self.avgpool(tcn_output)
        concat = torch.concatenate([avgpool, maxpool], dim=2)

        if explain:
            context, att_ws = self.attention(concat, explain)
        else:
            context = self.attention(concat)
        
        flat = self.flatten(context)
        out = self.sigmoid(self.fc(flat))
        
        if explain:
            return out, att_ws
        else:
            return out
    
    def call(self, x, explain=False):
        return self.forward(x, explain)

    def learn(self, Xdata, Ydata, batch_size, epochs):
        n_epochs = epochs    # number of epochs to run
        batch_size = batch_size  # size of each batch
        batches_per_epoch = len(Xdata) // batch_size

        avg_total_loss = []
        for epoch in tqdm(list(range(n_epochs)), desc="Epoch Training: "):
            # start_time = time.time()
            avg_loss = []
            for i in range(batches_per_epoch):
                start = i * batch_size
                # take a batch
                Xbatch = Xdata[start:start+batch_size]
                ybatch = Ydata[start:start+batch_size]
                # forward pass
                y_pred = self.forward(Xbatch)
                loss = self.loss_fn(y_pred, ybatch)
                avg_loss.append(loss.detach().numpy())
                avg_total_loss.append(loss.detach().numpy())
                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                # update weights
                self.optimizer.step()

            # finish_time = time.time()
            # print(f"Epoch {epoch}/{epochs}: {(finish_time-start_time):.2f}s;  Average_Epoch_Loss:{np.mean(avg_loss):.5f}, Total_Avg_Loss:{np.mean(avg_total_loss):.5f}")
        print(f"Total_Avg_Loss:{np.mean(avg_total_loss):.5f}")

