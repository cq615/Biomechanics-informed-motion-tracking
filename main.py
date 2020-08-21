import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from network import *
from dataio import *
from util import *
import time


n_class = 4
lr = 1e-4
n_worker = 4
bs = 1
n_epoch = 500
base_err = 100

model_load_path = './models/registration_model_pretrained_0.001_32.pth'
model_save_path = './models/registration_model.pth'
VAE_model_load_path = './models/VAE_recon_model_pretrained.pth'

# build and load registration and regularisation models
model = Registration_Net()
model.load_state_dict(torch.load(model_load_path))
model = model.cuda()
VAE_model = MotionVAE2D(img_size=96, z_dim=32)
VAE_model = VAE_model.cuda()
VAE_model.load_state_dict(torch.load(VAE_model_load_path))

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
flow_criterion = nn.MSELoss()
Tensor = torch.cuda.FloatTensor


def train(epoch):
    model.train()
    epoch_loss = []
    VAE_epoch_loss = []
    for batch_idx, batch in tqdm(enumerate(training_data_loader, 1),
                                 total=len(training_data_loader)):
        x, x_pred, x_gnd, mask = batch

        x_c = Variable(x.type(Tensor))
        x_predc = Variable(x_pred.type(Tensor))
        mask = Variable(mask.type(Tensor))

        net = model(x_c, x_predc, x_c)

        optimizer.zero_grad()
        max_norm = 0.1
        df_gradient = compute_gradient(net['out'])
        recon, mu, logvar = VAE_model(df_gradient, mask, max_norm)

        VAE_loss = MotionVAELoss(recon, df_gradient*mask, mu, logvar, beta=1e-4)
        loss = flow_criterion(net['fr_st'], x_predc) + 0.001 * VAE_loss

        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
        VAE_epoch_loss.append(VAE_loss.item())

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, VAE_Loss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(training_data_loader.dataset),
                100. * batch_idx / len(training_data_loader), np.mean(epoch_loss), np.mean(VAE_epoch_loss)))


def test():
    model.eval()
    test_loss = []
    VAE_test_loss = []
    global base_err
    for batch_idx, batch in tqdm(enumerate(testing_data_loader, 1),
                                 total=len(testing_data_loader)):
        x, x_pred, x_gnd, mask = batch
        x_c = Variable(x.type(Tensor))
        x_predc = Variable(x_pred.type(Tensor))
        mask = Variable(mask.type(Tensor))

        net = model(x_c, x_predc, x_c)

        max_norm = 0.1
        df_gradient = compute_gradient(net['out'])
        recon, mu, logvar = VAE_model(df_gradient, mask, max_norm)

        VAE_loss = MotionVAELoss(recon, df_gradient*mask, mu, logvar, beta=1e-4)

        loss = flow_criterion(net['fr_st'], x_predc) + 0.001*VAE_loss
        test_loss.append(loss.item())
        VAE_test_loss.append(VAE_loss.item())

    print('Loss: {:.6f}, VAE_Loss: {:.6f}'.format(np.mean(test_loss), np.mean(VAE_test_loss)))

    if np.mean(test_loss) < base_err:
        torch.save(model.state_dict(), model_save_path)
        print("Checkpoint saved to {}".format(model_save_path))
        base_err = np.mean(test_loss)


data_path = './data/cardiac_data/train'
train_set = TrainDataset(data_path)

test_data_path = './data/cardiac_data/val'
test_set = ValDataset(test_data_path)

# loading the data
training_data_loader = DataLoader(dataset=train_set, num_workers=n_worker,
                                  batch_size=bs, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=n_worker,
                                  batch_size=bs, shuffle=False)

for epoch in range(0, n_epoch + 1):
    start = time.time()
    train(epoch)
    end = time.time()
    print("training took {:.8f}".format(end-start))

    print('Epoch {}'.format(epoch))
    start = time.time()
    test()
    end = time.time()
    print("testing took {:.8f}".format(end-start))

