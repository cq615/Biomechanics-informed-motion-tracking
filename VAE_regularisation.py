import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

from network import *
from dataio import *
from util import *
import time


n_class = 4
lr = 1e-4
n_worker = 4
bs = 1
n_epoch = 1000
img_size = 96
max_norm = 0.1
base_err = 1000

model_load_path = './models/VAE_recon_model_pretrained.pth'
model_save_path = './models/VAE_recon_model.pth'


VAE_model = MotionVAE2D(img_size=96, z_dim=32)
VAE_model = VAE_model.cuda()
VAE_model.load_state_dict(torch.load(model_load_path))
optimizer = optim.Adam(filter(lambda p: p.requires_grad, VAE_model.parameters()), lr=lr)

Tensor = torch.cuda.FloatTensor


def train(epoch):
    VAE_model.train()
    epoch_loss = []
    for batch_idx, batch in tqdm(enumerate(training_data_loader, 1),
                                 total=len(training_data_loader)):
        disp, mask = batch

        disp = Variable(disp.type(Tensor))
        mask = Variable(mask.type(Tensor))

        optimizer.zero_grad()

        df_gradient = compute_gradient(disp)

        recon, mu, logvar = VAE_model(df_gradient, mask, max_norm)

        loss = MotionVAELoss(recon, df_gradient*mask, mu, logvar, beta=1e-4)

        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(disp), len(training_data_loader.dataset),
        100. * batch_idx / len(training_data_loader), np.mean(epoch_loss)))


def test():
    VAE_model.eval()
    test_loss = []
    global base_err
    for batch_idx, batch in tqdm(enumerate(testing_data_loader, 1),
                                 total=len(testing_data_loader)):
        disp, mask = batch
        disp = Variable(disp.type(Tensor))
        mask = Variable(mask.type(Tensor))

        df_gradient = compute_gradient(disp)

        recon, mu, logvar = VAE_model(df_gradient, mask, max_norm)

        loss = MotionVAELoss(recon, df_gradient*mask, mu, logvar, beta=1e-4)

        test_loss.append(loss.item())

    print('Base Loss: {:.6f}'.format(base_err))
    print('Test Loss: {:.6f}'.format(np.mean(test_loss)))

    if np.mean(test_loss) < base_err:
        torch.save(VAE_model.state_dict(), model_save_path)
        print("Checkpoint saved to {}".format(model_save_path))
        base_err = np.mean(test_loss)


data_path = './data/SimMotion'
train_set = TrainDataset_motion(data_path, 'train')
test_set = TrainDataset_motion(data_path, 'val')

# loading the data
training_data_loader = DataLoader(dataset=train_set, num_workers=n_worker,
                                  batch_size=bs, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=n_worker,
                                  batch_size=bs, shuffle=False)

for epoch in range(0, n_epoch + 1):

    print('Epoch {}'.format(epoch))

    start = time.time()
    train(epoch)
    end = time.time()
    print("training took {:.8f}".format(end-start))

    start = time.time()
    test()
    end = time.time()
    print("testing took {:.8f}".format(end - start))



