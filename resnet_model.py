import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from torch import nn
import torch
import ptwt
from copy import deepcopy


class Preprocessing(nn.Module):
    def __init__(self, normalize, remove_baseline, remove_hf_noise, result_only=True,
                 should_pre_transpose=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transforms = []
        self.result_only = result_only
        self.should_pre_transpose = should_pre_transpose

        if remove_baseline:
            self.transforms.append(self.f_remove_baseline)

        if remove_hf_noise:
            self.transforms.append(self.f_remove_hf_noise)

        if normalize:
            self.transforms.append(self.f_normalize)

    @staticmethod
    def threshold_fn(x, threshold):
        x[x < threshold] = 0
        return x

        # N = torch.as_tensor(x.clone().detach().shape[-1:], dtype=torch.float32).view(-1)
        #
        # # Compute the soft threshold
        # lambda_value = threshold * torch.sqrt(torch.tensor(2.0 * torch.log(N)))
        # soft_threshold = lambda_value / N
        #
        # # Apply the soft threshold to the input tensor
        # soft_thresholded = torch.sign(x) * torch.max(torch.abs(x) - soft_threshold, torch.tensor(0.0))
        #
        # return soft_thresholded

    @classmethod
    def f_remove_hf_noise(cls, signal):
        level = 4
        wavelet = 'db4'
        initial_signal = signal.reshape(-1, signal.shape[-1])

        coeffs = ptwt.wavedec(initial_signal, wavelet, level=level)

        # Estimate noise standard deviation using MAD-based method
        sigma = torch.median(torch.abs(coeffs[-level])) / 0.6745

        # Apply soft thresholding to coefficients
        coeffs[1:] = [cls.threshold_fn(c, sigma) for c in coeffs[1:]]

        # Reconstruct denoised signal using inverse wavelet transform
        denoised_signal = ptwt.waverec(coeffs, wavelet)

        return denoised_signal.reshape(*signal.shape)

    @classmethod
    def calculate_baseline(cls, signal):
        initial_signal = signal.detach().clone().reshape(-1, signal.shape[-1])

        ssd_shape = (initial_signal.shape[0], )

        generations = [{
            'signal': initial_signal,
            'mask': torch.ones(initial_signal.shape[0], dtype=torch.bool, device=signal.device)
        }]

        current_iter = 0

        while True:
            sig = generations[-1]['signal']
            lp, hp = ptwt.wavedec(sig, 'db4', level=1)
            new_ssd = torch.zeros(ssd_shape, device=signal.device)
            new_ssd[generations[-1]['mask']] = torch.sum(hp ** 2, dim=-1)
            generations[-1]['ssd'] = new_ssd

            if len(generations) >= 3:
                newly_stopped = torch.logical_and(
                    torch.gt(
                        generations[-3]['ssd'][generations[-1]['mask']],
                        generations[-2]['ssd'][generations[-1]['mask']],
                    ),
                    torch.lt(
                        generations[-2]['ssd'][generations[-1]['mask']],
                        generations[-1]['ssd'][generations[-1]['mask']],
                    ),
                )

                if torch.all(newly_stopped) or lp.shape[-1] < 8 or current_iter > 7:
                    break

                new_sig = lp[~newly_stopped]
                new_mask = torch.clone(generations[-1]['mask'])
                new_mask[generations[-1]['mask']] = ~newly_stopped

                generations.append({
                    'signal': new_sig,
                    'mask': new_mask
                })
            else:
                generations.append({
                    'signal': lp,
                    'mask': generations[-1]['mask']
                })

            current_iter += 1

        # for i in range(len(generations)):
        #     g = generations[i]
        #     plt.scatter([i for v in range(len(g['ssd']))], [v for v in g['ssd']], c=[v for v in range(len(g['ssd']))])
        #
        # plt.show()
        # exit()

        for i in range(len(generations) - 2, -1, -1):
            lp = generations[i + 1]['signal']
            sig = generations[i]['signal']

            recovered = ptwt.waverec([lp, torch.zeros_like(lp)], 'db4')
            if recovered.shape[-1] == sig.shape[-1] + 1:
                recovered = recovered[..., :-1]

            mask_diff = generations[i + 1]['mask'][generations[i]['mask']]  # massive tricks omg (or maskive?)
            sig[mask_diff] = recovered
        baseline = generations[0]['signal'].reshape(*signal.shape)
        return baseline

    @classmethod
    def f_remove_baseline(cls, x):
        bl = cls.calculate_baseline(x)

        return x - bl

    @classmethod
    def f_normalize(cls, x):
        std, mean = torch.std_mean(x, dim=-1, keepdim=True)
        x = torch.nan_to_num((x - mean) / std)
        return x, std, mean

    def forward(self, x):
        aux = []

        if self.should_pre_transpose:
            x = x.transpose(-1, -2)

        # plt.plot(x[0, 7, :2000], linewidth=0.5, color='red')
        # plt.savefig("pp_1.pdf")
        # plt.clf()
        # i = 2
        for t in self.transforms:
            output = t(x)
            if isinstance(output, tuple):
                x, *other = output
                aux += other
            else:
                x = output

            # plt.plot(x[0, 7, :2000], linewidth=0.5, color='red')
            # plt.savefig(f"pp_{i}.pdf")
            # plt.clf()
            # i += 1

        if self.should_pre_transpose:
            x = x.transpose(-1, -2)

        if self.result_only:
            return x
        else:
            return x, *aux
        

class ResidualUnit(nn.Module):
    def __init__(self, n_samples_out, n_filters_out, n_samples_in, n_filters_in, kernel_initializer='he_normal',
                 dropout_keep_prob=0.8, kernel_size=17, preactivation=True,
                 postactivation_bn=False, activation_function='gelu'):
        super(ResidualUnit, self).__init__()
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = 1 - dropout_keep_prob
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function
        self._gen_layer(n_samples_in, n_filters_in)

    def _skip_connection(self, downsample, n_filters_in):
        """Implement skip connection."""
        # Deal with downsampling
        layers = []

        if downsample > 1:
            layers.append(nn.MaxPool1d(downsample, padding=int((downsample - 1) / 2)))
        elif downsample == 1:
            pass
        else:
            raise ValueError("Number of samples should always decrease.")
        # Deal with n_filters dimension increase
        if n_filters_in != self.n_filters_out:
            # This is one of the two alternatives presented in ResNet paper
            # Other option is to just fill the matrix with zeros.
            layers.append(
                nn.Conv1d(in_channels=n_filters_in, out_channels=self.n_filters_out, kernel_size=1, padding='same'))
        return nn.Sequential(*layers)

    def _gen_layer(self, n_samples_in, n_filters_in):
        downsample = n_samples_in // self.n_samples_out

        self.skip_layer = self._skip_connection(downsample, n_filters_in)
        self.layer_1 = nn.Sequential(nn.Conv1d(n_filters_in, self.n_filters_out, self.kernel_size,
                                               padding='same'),
                                     # TODO not too sure
                                     nn.BatchNorm1d(self.n_filters_out),
                                     nn.GELU(),
                                     nn.Dropout1d(p=self.dropout_rate)
                                     )
        self.layer_2 = nn.Sequential(
            nn.Conv1d(self.n_filters_out, self.n_filters_out, self.kernel_size, stride=downsample),
        )
        self.layer_3 = nn.Sequential(
            nn.BatchNorm1d(self.n_filters_out),
            nn.GELU(),
            nn.Dropout1d(p=self.dropout_rate)
        )

    def forward(self, x):
        z = self.layer_1(x[0])
        z = self.layer_2(z)
        y = self.skip_layer(x[1])
        y = y + z
        x = self.layer_3(y)

        return [x, y]


class ResNet(nn.Module):
    def __init__(self, normalize=False, propagate_normalization=False, remove_baseline=False, remove_hf_noise=False,
                 embedding_size=256):
        super().__init__()
        self.preprocessing = Preprocessing(normalize, remove_baseline, remove_hf_noise, result_only=True)

        if propagate_normalization and not normalize:
            raise ValueError("Propagation is only meaningful when normalizing.")
        self.propagate_normalization = propagate_normalization

        kernel_size = 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(12, 12, kernel_size=kernel_size, stride=1, padding="same"),
            nn.BatchNorm1d(12),
            nn.GELU())

        self.layer1 = ResidualUnit(1024, 128, n_filters_in=12, n_samples_in=4096, kernel_size=kernel_size)
        self.layer2 = ResidualUnit(256, 196, n_filters_in=128, n_samples_in=1024, kernel_size=kernel_size)
        self.layer3 = ResidualUnit(64, 256, n_filters_in=196, n_samples_in=256, kernel_size=kernel_size)
        self.layer4 = ResidualUnit(16, 320, n_filters_in=256, n_samples_in=64, kernel_size=kernel_size)
        self.flattening = nn.Flatten(start_dim=1, end_dim=2)
        self.dense = nn.Sequential(torch.nn.Linear(5120 + (12 * 2 if propagate_normalization else 0), 512),
                                   nn.GELU(),
                                   torch.nn.Linear(512, embedding_size),
                                   nn.GELU(),
                                   torch.nn.Linear(embedding_size, embedding_size))

    def forward(self, x):
        x = self.preprocessing(x)

        x = self.conv1(x)
        x, y = self.layer1([x, x])
        x, y = self.layer2([x, y])
        x, y = self.layer3([x, y])
        x, _ = self.layer4([x, y])
        x = self.flattening(x)

        if self.propagate_normalization:
            raise NotImplementedError("Unsupported.")
            # x = torch.concat([x, mean.squeeze(), std.squeeze()], dim=1)

        x = self.dense(x)
        return x


if __name__ == "__main__":
    model = ResNet(normalize=True, propagate_normalization=False)
    X = model.forward(torch.normal(4, 15, size=(8, 12, 4096)))
    print(X.shape)
    print("done")