import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
from models.baselines import MLForecastModel

class BaseLinearModel(nn.Module):
    def __init__(self, args):
        super(BaseLinearModel, self).__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.channels = args.channels
        self.individual = args.individual if hasattr(args, 'individual') else False
        self.setup_layers()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def setup_layers(self):
        pass

    def forward(self, x):
        pass

    def fit(self, x, y):
        x ,y= x.float(), y.float()
        self.train()
        self.optimizer.zero_grad()
        outputs = self.forward(x)
        loss = self.calculate_loss(outputs, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def calculate_loss(self, outputs, y):
        if self.individual:
            return sum([self.criterion(outputs[:, :, i], y[:, :, i]) for i in range(self.channels)]) / self.channels
        else:
            return self.criterion(outputs, y)

    def forecast(self, x):
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions


class Linear_NN(BaseLinearModel):
    def setup_layers(self):
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        x = x.float()
        if self.individual:
            outputs = torch.zeros(x.size(0), self.pred_len, self.channels, dtype=x.dtype, device=x.device)
            for i, linear in enumerate(self.Linear):
                outputs[:, :, i] = linear(x[:, :, i])
        else:
            x = x.view(x.size(0), -1)  # Flatten the input
            outputs = self.Linear(x)
            outputs = outputs.view(x.size(0), self.pred_len, 1)
        return outputs


class DLinear(BaseLinearModel):
    def __init__(self, args):
        super().__init__()
        self.model = Model(args)

    def setup_layers(self):
        # TODO
        raise NotImplementedError

    def forward(self, x):
        # TODO
        raise NotImplementedError







