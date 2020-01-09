import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm


class Generator(nn.Module):
    def __init__(self, z_dim=128, img_channel=3, nf=64):
        """ Defines the generator model.

        Args:
            z_dim: The length of input random noise.
            img_channel: The number of channels of output image. 3 for CIFAR10, 1 for MNIST.
            nf: The unit number of filters. Default is 64.

        Returns:
            None.
        """
        super(Generator, self).__init__()
        self.nf = nf
        self.z_dim = z_dim

        self.latent = nn.Sequential(nn.Linear(self.z_dim, 8*nf*4*4),
                                    nn.BatchNorm1d(8*nf*4*4),
                                    nn.ReLU()
                                    )
    
        self.network = nn.Sequential(nn.ConvTranspose2d(nf*8, nf*4, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(nf*4),
                                     nn.ReLU(inplace=True),

                                     nn.ConvTranspose2d(nf*4, nf*2, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(nf*2),
                                     nn.ReLU(inplace=True),

                                     nn.ConvTranspose2d(nf*2, nf, 4, 2, 1, bias=False),
                                     nn.BatchNorm2d(nf),
                                     nn.ReLU(inplace=True),

                                     nn.ConvTranspose2d(nf, img_channel, kernel_size=1, stride=1, padding=0, bias=False),
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


class Discriminator(nn.Module):
    def __init__(self, len_code, img_channel=3, nf=64):
        """ Defines the discriminator and encoder model.

        Args:
            len_code: The length of output hash code.
            img_channel: The number of channels of input image. 3 for CIFAR10, 1 for MNIST.
            nf: The unit number of filters. Default is 64.

        Returns:
            None.
        """
        super(Discriminator, self).__init__()
        self.len_code = len_code
        self.nf = nf

        self.network = nn.Sequential(nn.Conv2d(img_channel, nf, 4, 2, 1),
                                     nn.LeakyReLU(2e-1),

                                     nn.Conv2d(nf, nf*2, 4, 2, 1),
                                     nn.BatchNorm2d(nf*2),
                                     nn.LeakyReLU(2e-1),
                                     
                                     nn.Conv2d(nf*2, nf*4, 4, 2, 1),
                                     nn.BatchNorm2d(nf*4),
                                     nn.LeakyReLU(2e-1),
                                     
                                     nn.Conv2d(nf*4, nf*8, 4, 2, 1),
                                     nn.BatchNorm2d(nf*8),
                                     nn.LeakyReLU(2e-1))

        self.discriminate = nn.Linear(8*nf*2*2, 1)
        self.encode = nn.Linear(8*nf*2*2, len_code)
        
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