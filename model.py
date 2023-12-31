import torch
import torch.nn as nn
from numpy import ndarray

import sys

import matplotlib.pyplot as plt
import torchvision.ops
from torch.nn.utils.weight_norm import weight_norm
from tqdm import tqdm
import pytorch_lightning as pl
import torchmetrics
from torch.nn.functional import one_hot


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, pad_mode):
        super(DeformableConv2d, self).__init__()

        self.padding = (padding, 0)
        self.dilation = (dilation, 1)
        self.ks = (kernel_size, 1)

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, self.ks, padding=self.padding,
                                     dilation=self.dilation, padding_mode=pad_mode, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels, kernel_size, self.ks, padding=self.padding, dilation=self.dilation,
                                        padding_mode=pad_mode, bias=True)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels, out_channels, self.ks, padding=self.padding, dilation=self.dilation,
                                      padding_mode=pad_mode, bias=False)

    def forward(self, x):
        h, w = x.shape[2:]
        max_offset = max(h, w) / 4.
        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(input=x, offset=offset, weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias, padding=self.padding, dilation=self.dilation,
                                          mask=modulator)
        return x


# One Conv. block
class Block(nn.Module):
    def __init__(self, model, c_in, c_out, ks, pad, dil, deformable):
        super(Block, self).__init__()
        self.model = model
        self.deform = deformable

        pad_mode = 'circular'

        if self.deform:
            self.conv = DeformableConv2d(c_in, c_out, ks, pad, dil, pad_mode)
        else:
            self.conv = weight_norm(nn.Conv1d(c_in, c_out, ks, padding=pad, dilation=dil, padding_mode=pad_mode))
            self.conv.weight.data.normal_(0, 0.01)
            self.conv.bias.data.normal_(0, 0.01)

        self.res = nn.Conv1d(c_in, c_out, kernel_size=(1,)) if c_in != c_out else None
        if self.res is not None:
            self.res.weight.data.normal_(0, 0.01)
            self.res.bias.data.normal_(0, 0.01)

        self.nonlinear = nn.GELU()

    def forward(self, x):
        net = self.conv

        if self.deform:
            x_2d = x.unsqueeze(-1)
            out = net(x_2d)
            res = x if self.res is None else self.res(x)
            y = self.nonlinear(out) + res.unsqueeze(-1)
            return y.squeeze(-1)
        else:
            out = net(x)
            res = x if self.res is None else self.res(x)
            return self.nonlinear(out) + res


# Conv. blocks
class ConvPart(nn.Module):
    def __init__(self, model, hidden_channels, ks):
        super(ConvPart, self).__init__()
        layers = []
        num_layer = len(hidden_channels)
        for i in range(1, num_layer):
            this_in = hidden_channels[i - 1]
            this_out = hidden_channels[i]
            this_dilation = 2 ** i
            this_padding = int(this_dilation * (ks - 1) / 2)
            if i < (num_layer - 3):
                layers += [Block(model, this_in, this_out, ks, this_padding, this_dilation, False)]
            else:
                layers += [Block(model, this_in, this_out, ks, this_padding, this_dilation, True)]
        self.conv_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_net(x)


# Conv. + classifier
class CDIL(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_channels=32,
                 output_channels=256,
                 num_layers=4,
                 kernel_size=3,
                 ):
        super(CDIL, self).__init__()

        channels = [hidden_channels] * num_layers
        channels[0] = input_channels
        channels[-1] = output_channels

        self.conv = ConvPart('dict-cdil', channels, kernel_size)

    def forward(self, x):
        for elem in x:
            std, mean = torch.std_mean(elem, dim=-1, keepdim=True)
            elem[:] = torch.nan_to_num((elem - mean) / std)

        return self.conv(x)


class AutoEncoder(pl.LightningModule):
    def __init__(self, model, lr, wd):
        super(AutoEncoder, self).__init__()
        self.model = model
        self.loss = torch.nn.MSELoss()
        self.lr = lr
        self.wd = wd

    def forward(self, x):
        return self.model(x.transpose(1, 2)).transpose(1, 2)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer
    
    
class Predictor(pl.LightningModule):
    def __init__(self, model, config):
        super(Predictor, self).__init__()
        self.model_type = config['model']
        self.model = model
        channels = config['output_channels']
        dropout = config['dropout']
        self.ignore_ecg = config['ignore_ecg']
        self.added_features = config['added_features']
        self.added_features_onehot = config['added_features_onehot']
        
        size = 0
        if not self.ignore_ecg:
            size += channels
        if config['added_features']:
            size += 3 + 10 # (reason must be one-hot encoded)
        if config['added_features_onehot']:
            size += 30
        
        self.MLPhead = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(size, size//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(size//2, size//4),
            nn.GELU(),
            nn.Linear(size//4, 3), 
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=3)
        self.lr = config['learning_rate']
        self.wd = config['weight_decay']
        self.ignore_ecg = config['ignore_ecg']
        
    @staticmethod
    def float_to_one_hot(x: ndarray, min_value, max_value, num_classes):
        x[x < min_value] = min_value
        x[x > max_value] = max_value
        x = (x - min_value) / (max_value - min_value) * (num_classes - 1)
        x = x.long()
        return one_hot(x, num_classes=num_classes)

    def forward(self, batch):
        x = batch['ecg']
        reasons = batch['visit_reasons']
        reasons_vec = one_hot(reasons, num_classes=10) # 10
        age = batch['ages'] # 1
        age_vec = self.float_to_one_hot(age, 18, 90, 10) # 10
        atrial_rate = batch['atrial_rates'] # 1
        atrial_rate_vec = self.float_to_one_hot(atrial_rate, 40, 180, 10) # 10
        ventricular_rate = batch['ventricular_rates'] # 1
        ventricular_rate_vec = self.float_to_one_hot(ventricular_rate, 40, 180, 10) # 10
        
        to_cat = []
        
        if not self.ignore_ecg:
            if self.model_type == 'cdil':
                model_out = self.model(x.transpose(1, 2)).transpose(1, 2)
                pooled = torch.mean(model_out, dim=1)
            elif self.model_type == 'resnet':
                pooled = self.model(x.transpose(1, 2))
            elif self.model_type == 'tsai01':
                pooled = self.model(x.transpose(1, 2))
                
            to_cat.append(pooled)
            # pooled = torch.cat((pooled, reasons_vec, age.view(-1, 1), age_vec, atrial_rate.view(-1, 1), atrial_rate_vec, ventricular_rate.view(-1, 1), ventricular_rate_vec), dim=1)
        
        if self.added_features:
            to_cat.append(reasons_vec)
            to_cat.append(age.view(-1, 1))
            to_cat.append(atrial_rate.view(-1, 1))
            to_cat.append(ventricular_rate.view(-1, 1))
            
        if self.added_features_onehot:
            to_cat.append(age_vec)
            to_cat.append(atrial_rate_vec)
            to_cat.append(ventricular_rate_vec)
            
        concat = torch.cat(to_cat, dim=1)
        concat = concat.float()
        output = self.MLPhead(concat)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss)
        self.log('train_accuracy', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss)
        self.log('val_accuracy', acc)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss)
        self.log('test_accuracy', acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer