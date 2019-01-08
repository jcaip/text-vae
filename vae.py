import os

import numpy as np
import torch
import torchvision.datasets as dset

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

import matplotlib.pyplot as plt

print(pyro.__version__)
pyro.enable_validation(True)
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)

cuda_enabled = True

def setup_data_loaders(batch_size=128, use_cuda=cuda_enabled):
    root = './data'

    trans = transforms.ToTensor()

    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
    test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=use_cuda)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=use_cuda)

    return train_loader, test_loader


class Decoder(nn.Module):

    def __init__(self, z_dim, hidden_dim):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 784)

        self.softplus = nn.Softplus()


    def forward(self, z):
        hidden = self.softplus(self.fc1(z))
        loc_img = torch.sigmoid(self.fc21(hidden))
        return loc_img

class Encoder(nn.Module):

    def __init__(self, z_dim, hidden_dim):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        x = x.reshape(-1, 784)
        hidden = self.softplus(self.fc1(x))
        z_loc =  self.fc21(hidden)
        z_scale =  torch.exp(self.fc22(hidden))
        return z_loc, z_scale

class VAE(nn.Module):

    def __init__(self, z_dim=50, hidden_dim=400, use_cuda=cuda_enabled):
        super(VAE, self).__init__()

        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        if use_cuda:
            self.cuda()

        self.use_cuda = use_cuda
        self.z_dim = z_dim

    def model(self, x):

        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))

            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

            loc_img = self.decoder.forward(z)
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))
            return loc_img


    def guide(self, x):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder.forward(x)
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def reconstruct_img(self, x):
        z_loc, z_scale = self.encoder(x)
        z = dist.Normal(z_loc, z_scale).sample()
        loc_img = self.decoder(z)
        return loc_img

pyro.clear_param_store()

vae = VAE(use_cuda=cuda_enabled)
optimizer = Adam({'lr': 1.0e-3})
svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

NUM_EPOCHS = 10
TEST_FREQUENCY = 5
print("Getting data")
train_loader, test_loader = setup_data_loaders(batch_size=256, use_cuda=cuda_enabled)
print("Starting training")

train_elbo, test_elbo = [], []

for epoch in range(NUM_EPOCHS):

    epoch_loss = 0.
    for x, _ in train_loader:
        if cuda_enabled:
            x = x.cuda()
        epoch_loss += svi.step(x)

    total_epoch_loss_train = epoch_loss / len(train_loader.dataset)
    train_elbo.append(-total_epoch_loss_train)
    print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

    # report test diagnostics
    test_loss = 0.
    for i, (x, _) in enumerate(test_loader):
        if cuda_enabled:
            x = x.cuda()
        test_loss += svi.evaluate_loss(x)

        if i == 0:
            for index in [12, 18, 55]:
                test_img = x[index, :]
                f, axarr = plt.subplots(1, 2)
                reco_img = vae.reconstruct_img(test_img)
                test_img = test_img.view(28, 28)
                reco_img = reco_img.view(28, 28)
                axarr[0].imshow(test_img.detach().cpu())
                axarr[1].imshow(reco_img.detach().cpu())
                plt.savefig("epoch_{}_reconstruction.png".format(epoch))



    total_epoch_loss_test = test_loss / len(test_loader.dataset)
    test_elbo.append(-total_epoch_loss_test)
    print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))

