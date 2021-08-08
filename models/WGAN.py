import matplotlib, random
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.optim import Adam
import numpy as np
from matplotlib import pyplot as plt
import torchvision.utils as vutils

def visualize(batch, save=None):
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(batch, padding=2, normalize=True).cpu(),(1,2,0)))
    if save is None: plt.show()
    if save is not None:
        plt.savefig(save)
    plt.close()

class Critic(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Critic, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def gradient_penalty(critic, real, fake, device="cuda"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

class WGANGP(nn.Module):
    def __init__(self):
        super(WGANGP, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gen = Generator(100, 3, 64).to(device)
        self.critic = Critic(3, 64).to(device)
        # initialize_weights(self.gen)
        # initialize_weights(self.critic)

    def train(self, OUTPATH):
        critic = self.critic
        gen = self.gen

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        LATENT_DIM = 100
        CRITIC_ITERS = 5
        BATCH_SIZE = 64
        NUM_EPOCHS = 200
        LR = 0.0001
        LAMBDA_GP = 10
        DATAPATH = 'D:\\repos\\NekoNet\\data\\cropped224'


        fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)
        visualize(gen(fixed_noise), f'{OUTPATH}/visual/92.jpg')

        dataset = ImageFolder(root=DATAPATH,
                            transform=T.Compose([
                            T.Resize((64, 64)),
                            T.ToTensor(),
                            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                                shuffle=True, num_workers=2)

        # Setup Adam optimizers for both G and D
        optC = Adam(critic.parameters(), LR, betas=(0, 0.9))
        optG = Adam(gen.parameters(), LR, betas=(0, 0.9))


        iters = 0
        for epoch in range(NUM_EPOCHS):
            for _, (real, _) in enumerate(dataloader):
                iters += 1
                real = real.to(device)
                cur_batch_size = real.shape[0]

                # Critic: max E[critic(real)] - E[critic(fake)]
                for _ in range(CRITIC_ITERS):
                    noise = torch.randn((cur_batch_size, LATENT_DIM, 1, 1), device=device)
                    fake = gen(noise)
                    critic_real = critic(real).reshape(-1)
                    critic_fake = critic(fake).reshape(-1)
                    gp = gradient_penalty(critic, real, fake)
                    loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP*gp
                    critic.zero_grad()
                    loss_critic.backward()
                    optC.step()

                # Generator: max E[critic(fake)]
                fake = gen(noise)
                critic_fake = critic(fake).reshape(-1)
                loss_gen = -torch.mean(critic_fake)
                gen.zero_grad()
                loss_gen.backward()
                optG.step()

                if iters%500 == 0 or (iters < 1000 and iters%100==0):
                    gen.eval()
                    visualize(gen(fixed_noise), f'{OUTPATH}/visual/{iters}.jpg')
                    gen.train()

            state = {
                'G': gen.state_dict(),
                'D': critic.state_dict(),
                'opG': optG.state_dict(),
                'opC': optC.state_dict()
            }
            torch.save(state, 'state.pt')