import sys
from pathlib import Path
import argparse
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch.autograd import grad, Variable
import torch.backends.cudnn as cudnn

from models import Generator, Discriminator
from utils import get_trmat, set_input, get_prec_topn, bit_entropy


class HashGAN(object):
    def __init__(self, z_dim, b_dim, device, save_dir, dataset='mnist', G_dict=None, D_dict=None):
        """ Defines the HashGAN network.

        Args:
            z_dim: The length of a continuous part of input random variable. 
            b_dim: The length of a binary part of input random variable.
            device: The device number to use. It should be an integer value.
            save_dir: The path to a directory where the generated images and trained models will be saved.
            dataset: The name of dataset to use. 'cifar10' and 'mnist' are possible.
            G_dict: The path of generator model to load. If this is None, generator will be randomly initialized.
            D_dict: The path of discriminator(encoder) model to load. If this is None, discriminator(encoder) will be randomly initialized.

        Returns:
            None.
        """
        self.device = device
        self.z_dim = z_dim
        self.b_dim = b_dim
        self.input_dim = z_dim + b_dim
        self.dataset = dataset

        self.save_dir = Path('results')/dataset/save_dir
        self.model_dir = self.save_dir/'models'
        self.img_dir = self.save_dir/'images'    

        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.img_dir).mkdir(parents=True, exist_ok=True)

        if self.dataset == 'mnist':
            img_channel = 1
        else:
            img_channel = 3
        
        self.G = Generator(self.input_dim, img_channel).to(self.device)
        self.D = Discriminator(self.b_dim, img_channel).to(self.device)

        if G_dict:
            G_weight = torch.load(G_dict)
            G_dict = self.G.state_dict()
            G_weight = {k: v for k, v in G_weight.items() if k in G_dict}
            G_dict.update(G_weight)
            self.G.load_state_dict(G_dict)
        if D_dict: 
            D_weight = torch.load(D_dict)
            D_dict = self.D.state_dict()
            D_weight = {k: v for k, v in D_weight.items() if k in D_dict}
            D_dict.update(D_weight)
            self.D.load_state_dict(D_dict)

    def loss_D(self, x_real, hashing=True, zb_input=None):
        """ Computes the dicriminator and encoder losses and stores them as attributes of HashGAN class.

        Args:
            x_real: The real image to compute loss.
            hashing: Whether to compute hashing loss.
            zb_input: The input random noise to generate a fake image. If this is None, an input random noise will be set randomly.

        Returns:
            None.
        """
        bs = x_real.size(0)
        if zb_input is None:
            zb_input = set_input(bs, self.z_dim, self.b_dim, self.device)
        b_input = zb_input[:, -self.b_dim:]
        x_fake = self.G(zb_input).detach()
        fake_logits, fake_codes, fake_feats = self.D(x_fake)
        real_logits, real_codes, real_feats = self.D(x_real)

        self.adv_loss_real = nn.BCELoss()(real_logits, torch.ones(bs, 1).to(self.device))
        self.adv_loss_fake = nn.BCELoss()(fake_logits, torch.zeros(bs, 1).to(self.device))
        self.adv_loss = self.adv_loss_real + self.adv_loss_fake
        
        if hashing:  
            aff = get_trmat(bs, 10, 0.1).to(self.device)
            affgrid = F.affine_grid(aff, x_real.size())
            x_real_aff = F.grid_sample(x_real, affgrid, padding_mode='reflection')
            real_aff_codes = self.D(x_real_aff)[1]
            mean_codes = real_codes.mean(dim=0)

            self.min_entropy_loss = bit_entropy(real_codes, 'mean')
            self.uniform_freq_loss = - bit_entropy(mean_codes, 'mean')
            self.consistent_loss = (real_codes-real_aff_codes).pow(2).mean()
            self.independent_loss = ((self.D.encode.weight.T @ self.D.encode.weight)
                                - torch.eye(self.D.encode.weight.size(1)).to(self.device)).pow(2).mean()

            self.hash_loss = 0.01*self.min_entropy_loss + self.uniform_freq_loss + self.consistent_loss + self.independent_loss

            self.col_l2_loss = (fake_codes - b_input).pow(2).sum(dim=1).mean()
        else:
            self.min_entropy_loss = torch.Tensor([0.0]).to(self.device)
            self.uniform_freq_loss = torch.Tensor([0.0]).to(self.device)
            self.consistent_loss = torch.Tensor([0.0]).to(self.device)
            self.independent_loss = torch.Tensor([0.0]).to(self.device)
            self.hash_loss = torch.Tensor([0.0]).to(self.device)
            self.col_l2_loss = torch.Tensor([0.0]).to(self.device)


    def loss_G(self, x_real, zb_input=None):
        """ Computes the feature matching loss and stores it as an attribute of HashGAN class.

        Args:
            x_real: The real image to compute loss.
            zb_input: The input random noise to generate a fake image. If this is None, an input random noise will be set randomly.

        Returns:
            None.
        """
        bs = x_real.size(0)
        if zb_input is None:
            zb_input = set_input(bs, self.z_dim, self.b_dim, self.device)
        x_fake = self.G(zb_input)

        real_feats = self.D(x_real)[2]
        fake_feats = self.D(x_fake)[2]

        self.feat_match_loss = (real_feats.mean(dim=0)-fake_feats.mean(dim=0)).pow(2).mean()


    def step_opt(self, loss, opt, retain_graph=False):
        """ Computes gradient and step optimizer.

        Args:
            loss: The loss to compute gradient.
            opt: The optimizer to use.
            retain_graph: Whether to retain the graph.
        
        Returns:
            None.
        """
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        opt.step()

    def generate_code_label(self, dataloader):
        """ Generates codes and labels of all datapoints in given dataloader.

        Args:
            dataloader: The dataloader to get codes and labels, instance of torch.utils.data.DataLoader.
            
        Returns:
            Generated binary(-1 or 1) codes and corresponding one-hot labels.
        """  
        with torch.no_grad():
            bs = dataloader.batch_size
            num_data = len(dataloader.dataset)
            codes_all = torch.zeros([num_data, self.b_dim])
            labels_all = torch.zeros([num_data, 10])

            train_it = iter(dataloader)
            t_train = trange(0, len(dataloader), initial=0, total=len(dataloader))

            for step in t_train:
                index = torch.arange(step*bs, (step+1)*bs)
                try:
                    img, label = next(train_it)
                except StopIteration:
                    continue
                img = img.to(self.device)
                code = self.D(img)[1]
                codes_all[index, :] = (code-0.5).sign().cpu()
                labels_all[index, label] = 1
        return codes_all, labels_all

    def eval(self, query_loader, database_loader):
        """ Evaluates the hashgan network with given query set and database set.

        Args:
            query_loader: Dataloader of query set. Instance of torch.utils.data.DataLoader.
            database_loader: Dataloader of database set. Instance of torch.utils.data.DataLoader.

        Returns:
            Computed precision@1000 of given query set and database set.
        """
        self.G.eval()
        self.D.eval()
        query_code, query_label = self.generate_code_label(query_loader)
        database_code, database_label = self.generate_code_label(database_loader)
        mAP1000 = get_prec_topn(query_code, database_code, query_label, database_label, topn=1000)
        self.G.train()
        self.D.train()
        return mAP1000


    def train(self, data_loader, init_lr, final_lr, num_epoch, log_step, query_loader=None, database_loader=None):
        """ Train the network.

        Args:
            data_lodaer: Dataloader of training set. Instance of torch.utils.data.DataLoader.
            init_lr: The initial learning rate.
            final_lr: The final learning rate.
            num_epoch: Maximum epoch to train.
            log_step: Interval to print the losses.
            query_loader: Dataloader of query set to evaluate network during training. If this is None, the network will be not evaluated during training.
            database_loader: Dataloader of database set to evaluate network during training. If this is None, the network will be not evaluated during training.

        Returns:
            None.
        """
        cudnn.benchmark = True
        bs = data_loader.batch_size

        self.num_epoch = num_epoch
        self.G.train()
        self.D.train()

        self.G_opt = torch.optim.Adam(self.G.parameters(), lr=init_lr, betas=(0.5, 0.999))
        self.D_opt = torch.optim.Adam(self.D.parameters(), lr=init_lr, betas=(0.5, 0.999))

        z_example = set_input(bs, self.z_dim, self.b_dim, self.device)

        best_mAP1000 = 0.0

        for epoch in range(num_epoch):
            self.epoch = epoch
            train_it = iter(data_loader)
            t_train = trange(0, len(data_loader), initial=0,
                             total=len(data_loader))

            for step in t_train:
                try:
                    dp = next(train_it)
                except StopIteration:
                    continue

                x_real = dp[0]
                x_real = x_real.to(self.device)
                self.loss_G(x_real)
                self.step_opt(self.feat_match_loss, self.G_opt)
                if epoch < num_epoch/10:
                    self.loss_D(x_real, hashing=False)
                    self.step_opt(self.adv_loss, self.D_opt)

                else:
                    self.loss_D(x_real)
                    D_loss = self.adv_loss + self.hash_loss + 0.1*self.col_l2_loss
                    self.step_opt(D_loss, self.D_opt)

                if not (step) % log_step:
                    t_train.set_description(f'Epoch[{epoch}],'
                                            + f'G:[{self.feat_match_loss.item():.3f}],'
                                            + f'adv:[{self.adv_loss.item():.3f}],'
                                            + f'me:[{self.min_entropy_loss.item():.3f}],'
                                            + f'uf:[{self.uniform_freq_loss.item():.3f}],'
                                            + f'cons:[{self.consistent_loss.item():.3f}],'
                                            + f'W:[{self.independent_loss.item():.3f}],'
                                            + f'col:[{self.col_l2_loss.item():.3f}]'
                                            )
        
            if epoch == 0:
                vutils.save_image(x_real, self.img_dir/'real.png', normalize=True, nrow=int(np.sqrt(bs)))

            x_fake_example = self.G(z_example)
            vutils.save_image(x_fake_example.view(x_real.size()), 
                              self.img_dir/f'fake_epoch_{epoch}.png', 
                              normalize=True, 
                              nrow=int(np.sqrt(bs)))
            
            if query_loader and database_loader and not epoch%10:
                mAP1000 = self.eval(query_loader, database_loader)
                if mAP1000 >= best_mAP1000:
                    torch.save(self.D.state_dict(), self.model_dir/f'D_best.ckpt')
                    torch.save(self.G.state_dict(), self.model_dir/f'G_best.ckpt')
                    best_mAP1000 = mAP1000
                print(f'mAP_1000: {mAP1000:.4f}, best_mAP1000: {best_mAP1000:.4f}')

            if not (epoch+1) % (num_epoch//5):
                self.G_opt.param_groups[0]['lr'] += (final_lr-init_lr)/(4)
                self.D_opt.param_groups[0]['lr'] += (final_lr-init_lr)/(4)
                print(f'==== lr has changed to {self.G_opt.param_groups[0]["lr"]} ====')

            torch.save(self.D.state_dict(), self.model_dir/f'D_final.ckpt')
            torch.save(self.G.state_dict(), self.model_dir/f'G_final.ckpt')
                    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--batch_size', type=int,
                        default=100, help='The size of a batch')
    parser.add_argument('--z_dim', type=int, default=128,
                        help='The size of a latent vector')
    parser.add_argument('--b_dim', type=int, default=16,
                        help='The length of hash code')
    parser.add_argument('--init_lr', type=float, default=9e-4,
                        help='The initial learning rate')
    parser.add_argument('--final_lr', type=float,
                        default=3e-4, help='The final learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='The number of epochs to run')
    parser.add_argument('--gpu', type=int, default=0,
                        help='The ID of GPU to use')
    parser.add_argument('--dataset', type=str,
                        default='mnist', help='The dataset to use')
    parser.add_argument('--G_dict', type=str,
                        default=None, help='The path to generator model to load')
    parser.add_argument('--D_dict', type=str,
                        default=None, help='The path to discriminator model to load')
    parser.add_argument('--save_dir', type=str,
                        default='temp', help='The name of the save directory')
    parser.add_argument('--log_step', type=int,
                        default=10, help='Log step period')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='The images should be in here')

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    net = HashGAN(args.z_dim, args.b_dim, device,
                  args.save_dir, args.dataset, args.G_dict, args.D_dict)


    if args.dataset == 'mnist':
        img_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,)),
                                            ])
        dataset = datasets.MNIST(
            args.data_dir, train=True, download=True, transform=img_transforms)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                drop_last=True,
                                                pin_memory=True)
        query_set = datasets.MNIST(
            args.data_dir, train=False, download=True, transform=img_transforms)
        query_loader = torch.utils.data.DataLoader(dataset=query_set,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                drop_last=True,
                                                pin_memory=True)
        database_set = datasets.MNIST(
            args.data_dir, train=True, download=True, transform=img_transforms)
        database_loader = torch.utils.data.DataLoader(dataset=query_set,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    drop_last=True,
                                                    pin_memory=True)

    elif args.dataset == 'cifar10':
        img_transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(
                                            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ])
        dataset = datasets.CIFAR10(
            args.data_dir, train=True, download=True, transform=img_transforms)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                drop_last=True,
                                                pin_memory=True)
        query_set = datasets.CIFAR10(
            args.data_dir, train=False, download=True, transform=img_transforms)
        query_loader = torch.utils.data.DataLoader(dataset=query_set,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                drop_last=True,
                                                pin_memory=True)
        database_set = datasets.CIFAR10(
            args.data_dir, train=True, download=True, transform=img_transforms)
        database_loader = torch.utils.data.DataLoader(dataset=query_set,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    drop_last=True,
                                                    pin_memory=True)
    
    else:
        print('The given dataset is invalid!')
        sys.exit()

    if args.train:
        net.train(data_loader, args.init_lr, args.final_lr,
                args.epochs, args.log_step)
    elif args.eval:
        net.eval(query_loader, database_loader)