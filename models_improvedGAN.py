import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm


class G_cifar10(nn.Module):
        """ Defines the generator model.

        Args:
            z_dim: The length of input random noise.
            nf: The unit number of filters. Default is 64.

        Returns:
            None.
        """
    def __init__(self, z_dim=128, nf=64):
        super(G_cifar10, self).__init__()
        self.nf = nf
        self.z_dim = z_dim

        self.latent = nn.Sequential(nn.Linear(self.z_dim, 8*nf*4*4),
                                    nn.BatchNorm1d(8*nf*4*4),
                                    nn.ReLU()
                                    )
        self.network = nn.Sequential(nn.ConvTranspose2d(8*nf, 4*nf, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(4*nf),
                                     nn.ReLU(inplace=True),
                                     nn.ConvTranspose2d(4*nf, 2*nf, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(2*nf),
                                     nn.ReLU(inplace=True),
                                     weight_norm(nn.ConvTranspose2d(2*nf, 3, 4, 2, 1, bias=False)),
                                     nn.Tanh()
                                     )
        self._initialize_weights()
        
    def forward(self, z):
        out = self.latent(z)
        out = out.reshape(out.size(0), 8*self.nf, 4, 4)
        out = self.network(out)
        return out

    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class DE_cifar10(nn.Module):
        """ Defines the discriminator and encoder model.

        Args:
            len_code: The length of output hash code.
            nf: The unit number of filters. Default is 64.

        Returns:
            None.
        """
    def __init__(self, len_code, nf=48):
        super(DE_cifar10, self).__init__()
        self.len_code = len_code
        self.nf = nf

        self.network = nn.Sequential(nn.Dropout(0.2),
                                     weight_norm(nn.Conv2d(3, 2*nf, 3, 1, 1)),
                                     nn.LeakyReLU(2e-1),
                                     weight_norm(
                                         nn.Conv2d(2*nf, 2*nf, 3, 1, 1)),
                                     nn.LeakyReLU(2e-1),
                                     weight_norm(
                                         nn.Conv2d(2*nf, 2*nf, 3, 2, 1)),
                                     nn.LeakyReLU(2e-1),
                                     nn.Dropout(0.5),
                                     weight_norm(
                                         nn.Conv2d(2*nf, 4*nf, 3, 1, 1)),
                                     nn.LeakyReLU(2e-1),
                                     weight_norm(
                                         nn.Conv2d(4*nf, 4*nf, 3, 1, 1)),
                                     nn.LeakyReLU(2e-1),
                                     weight_norm(
                                         nn.Conv2d(4*nf, 4*nf, 3, 2, 1)),
                                     nn.LeakyReLU(2e-1),
                                     nn.Dropout(0.5),
                                     weight_norm(
                                         nn.Conv2d(4*nf, 4*nf, 3, 1, 0)),
                                     nn.LeakyReLU(2e-1),
                                     weight_norm(
                                         nn.Conv2d(4*nf, 4*nf, 1, 1, 0)),
                                     nn.LeakyReLU(2e-1),
                                     weight_norm(
                                         nn.Conv2d(4*nf, 4*nf, 1, 1, 0)),
                                     nn.LeakyReLU(2e-1),
                                     nn.AdaptiveAvgPool2d((1, 1)),
                                     )

        self.discriminate = weight_norm(nn.Linear(4*nf, 1))
        self.encode = nn.Linear(4*nf, len_code)
        
        self._initialize_weights()

    def forward(self, x):
        feat = self.network(x)
        feat = feat.view(feat.size(0), -1)
        disc = self.discriminate(feat)
        disc = nn.Sigmoid()(disc)
        code = self.encode(feat)
        code = nn.Sigmoid()(code)
        return disc, code, feat
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class G_mnist(nn.Module):
        """ Defines the generator model.

        Args:
            z_dim: The length of input random noise.
            nf: The unit number of filters. Default is 64.

        Returns:
            None.
        """
    def __init__(self, z_dim=128):
        super(G_mnist, self).__init__()
        self.z_dim = z_dim

        self.network = nn.Sequential(nn.Linear(self.z_dim, 500),
                                     nn.BatchNorm1d(500),
                                     nn.Softplus(),
                                     nn.Linear(500, 500),
                                     nn.BatchNorm1d(500),
                                     nn.Softplus(),
                                     weight_norm(nn.Linear(500, 28**2)),
                                     nn.Tanh(),
                                     )      
        self._initialize_weights()
        
    def forward(self, z):
        out = self.network(z)
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class DE_mnist(nn.Module):
        """ Defines the discriminator and encoder model.

        Args:
            len_code: The length of output hash code.
            nf: The unit number of filters. Default is 64.

        Returns:
            None.
        """
    def __init__(self, len_code):
        super(DE_mnist, self).__init__()
        self.len_code = len_code
        self.network = torch.nn.Sequential(weight_norm(nn.Linear(28**2, 1000)),
                                           nn.ReLU(),
                                           AddNoise(0.15),
                                           weight_norm(nn.Linear(1000, 500)),
                                           nn.ReLU(),
                                           AddNoise(0.15),
                                           weight_norm(nn.Linear(500, 250)),
                                           nn.ReLU(),
                                           AddNoise(0.15),
                                           weight_norm(nn.Linear(250, 250)),
                                           nn.ReLU(),
                                           AddNoise(0.15),
                                           weight_norm(nn.Linear(250, 250)),
                                           )

        self.discriminate = weight_norm(nn.Linear(250, 1))
        self.encode = nn.Linear(250, len_code)
        self._initialize_weights()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        feat = self.network(x)
        disc = self.discriminate(feat)
        disc = nn.Sigmoid()(disc)
        code = self.encode(feat)
        code = nn.Sigmoid()(code)
        return disc, code, feat
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class AddNoise(nn.Module):
    def __init__(self, std=0.05):
        super().__init__()
        self.noise = torch.Tensor([0])
        self.std = std

    def forward(self, x):
        device = torch.device(f'cuda:{x.get_device()}')
        x = x.cpu()
        if self.training and self.std != 0:
            scale = self.std * x
            noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + noise
        x = x.to(device)
        return x