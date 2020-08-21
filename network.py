import torch
from torch import nn
import torch.nn.functional as F
import time

# Basic model class including load and save wrap up
class BasicMoudule(nn.Module):

    def __init__(self):
        super(BasicMoudule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name


def relu():
    return nn.ReLU(inplace=True)


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding = 1, nonlinearity = relu):
    conv_layer = nn.Conv2d(in_channels = in_channels, out_channels= out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)

    nll_layer = nonlinearity()
    bn_layer = nn.BatchNorm2d(out_channels)

    layers = [conv_layer, bn_layer, nll_layer]
    return nn.Sequential(*layers)


def conv_blocks_2(in_channels, out_channels, strides=1):
    conv1 = conv(in_channels, out_channels, stride = strides)
    conv2 = conv(out_channels, out_channels, stride=1)
    layers = [conv1, conv2]
    return nn.Sequential(*layers)


def conv_blocks_3(in_channels, out_channels, strides=1):
    conv1 = conv(in_channels, out_channels, stride = strides)
    conv2 = conv(out_channels, out_channels, stride=1)
    conv3 = conv(out_channels, out_channels, stride=1)
    layers = [conv1, conv2, conv3]
    return nn.Sequential(*layers)


def transform(seg_source, loc, mode='bilinear'):
    grid = generate_grid(seg_source, loc)
    out = F.grid_sample(seg_source, grid, mode=mode)
    return out


def generate_grid(x, offset):
    x_shape = x.size()
    grid_w, grid_h = torch.meshgrid([torch.linspace(-1, 1, x_shape[2]), torch.linspace(-1, 1, x_shape[3])])  # (h, w)
    grid_w = grid_w.cuda().float()
    grid_h = grid_h.cuda().float()

    grid_w = nn.Parameter(grid_w, requires_grad=False)
    grid_h = nn.Parameter(grid_h, requires_grad=False)

    offset_h, offset_w = torch.split(offset, 1, 1)
    offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
    offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)

    offset_w = grid_w + offset_w
    offset_h = grid_h + offset_h

    offsets = torch.stack((offset_h, offset_w), 3)
    return offsets


class Registration_Net(nn.Module):
    """Deformable registration network with input from image space """
    def __init__(self, n_ch=1):
        super(Registration_Net, self).__init__()

        self.conv_blocks = [conv_blocks_2(n_ch, 64), conv_blocks_2(64, 128, 2), conv_blocks_3(128, 256, 2), conv_blocks_3(256, 512, 2), conv_blocks_3(512, 512, 2)]
        self.conv = []
        for in_filters in [128, 256, 512, 1024, 1024]:
            self.conv += [conv(in_filters, 64)]

        self.conv_blocks = nn.Sequential(*self.conv_blocks)
        self.conv = nn.Sequential(*self.conv)

        self.conv6 = nn.Conv2d(64 * 5, 64, 1)
        self.conv7 = conv(64, 64, 1, 1, 0)
        self.conv8 = nn.Conv2d(64, 2, 1)

    def forward(self, x, x_pred, x_img, mode='bilinear'):
        # x: source image; x_pred: target image; x_img: source image or segmentation map
        net = {}
        net['conv0'] = x
        net['conv0s'] = x_pred
        for i in range(5):
            net['conv%d'% (i+1)] = self.conv_blocks[i](net['conv%d'%i])
            net['conv%ds' % (i + 1)] = self.conv_blocks[i](net['conv%ds' % i])
            net['concat%d'%(i+1)] = torch.cat((net['conv%d'% (i+1)], net['conv%ds' % (i + 1)]), 1)
            net['out%d'%(i+1)] = self.conv[i](net['concat%d'%(i+1)])
            if i > 0:
                net['out%d_up'%(i+1)] = F.interpolate(net['out%d'%(i+1)], scale_factor=2**i, mode='bilinear', align_corners=True)

        net['concat'] = torch.cat((net['out1'], net['out2_up'], net['out3_up'], net['out4_up'], net['out5_up']), 1)
        net['comb_1'] = self.conv6(net['concat'])
        net['comb_2'] = self.conv7(net['comb_1'])

        net['out'] = torch.tanh(self.conv8(net['comb_2']))
        net['grid'] = generate_grid(x_img, net['out'])
        net['fr_st'] = F.grid_sample(x_img, net['grid'], mode=mode)

        return net


class MotionVAE2D(BasicMoudule):
    """VAE regularisation to reconstruct gradients of deformation fields """
    def __init__(self, img_size=80, z_dim=8, nf=32):
        super(MotionVAE2D, self).__init__()

        # input 1 x n x n
        self.conv1 = nn.Conv2d(4, nf, kernel_size=4, stride=2, padding=1)
        # size nf x n/2 x n/2
        self.conv2 = nn.Conv2d(nf, nf * 2, kernel_size=4, stride=2, padding=1)
        # size nf*2 x n/4 x n/4
        self.conv3 = nn.Conv2d(nf * 2, nf * 4, kernel_size=4, stride=2, padding=1)
        # size nf*4 x n/8 x n/8
        self.conv4 = nn.Conv2d(nf * 4, nf * 8, kernel_size=4, stride=2, padding=1)
        # size nf*8 x n/16*n/16

        h_dim = int(nf * 8 * img_size / 16 * img_size / 16)

        self.fc11 = nn.Linear(h_dim, z_dim)
        self.fc12 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(z_dim, h_dim)

        self.deconv1 = nn.ConvTranspose2d(nf * 8, nf * 4, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(nf * 4, nf * 2, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(nf * 2, nf, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(nf, 4, kernel_size=4, stride=2, padding=1)

        self.encoder = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            self.conv3,
            nn.ReLU(),
            self.conv4,
            nn.ReLU(),
            Flatten()
        )

        self.decoder = nn.Sequential(
            UnFlatten(C=int(nf * 8), H=int(img_size / 16), W=int(img_size / 16)),
            self.deconv1,
            nn.ReLU(),
            self.deconv2,
            nn.ReLU(),
            self.deconv3,
            nn.ReLU(),
            self.deconv4,
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def bottleneck(self, h):
        mu, logvar = self.fc11(h), self.fc12(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc2(z)
        z = self.decoder(z)
        return z

    def forward(self, x, mask, max_norm):
        # mask: dilated myocardial mask; max_norm: to scale data
        # only reconstruct myocardial motion
        x = x * mask
        x = x/max_norm
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z*max_norm, mu, logvar

# Flatten layer
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# UnFlatten layer
class UnFlatten(nn.Module):
    def __init__(self, C, H, W):
        super(UnFlatten, self).__init__()
        self.C, self.H, self.W = C, H, W

    def forward(self, input):
        return input.view(input.size(0), self.C, self.H, self.W)


def MotionVAELoss(recon_x, x, mu, logvar, beta=1e-2):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta*KLD