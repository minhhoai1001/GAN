import argparse
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm import tqdm
from model import Generator, Discriminator
from dataset import CelebaDataset
import config
from utils import Weights_Init, VisualLoss, Save_Checkpoint, RealvsFake
import random

def train(netD, netG, train_dataloader, train_dataset, optimizerD, optimizerG, criterion):
    print('Training')
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataset)/train_dataloader.batch_size))
    for i, data in prog_bar:
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data.to(config.device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=config.device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, config.nz, 1, 1, device=config.device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
    return errD, errG, D_x, D_G_z1, D_G_z2

def main(agrs):
    train_dataset = CelebaDataset(root=agrs.train_path, transform=config.train_transform)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.batch_size,
        num_workers=config.workers,
        pin_memory=True,
    )
    print(f"Number iteration: {len(train_loader)}")
    # Set random seed for reproducibility
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # 1. Create the generator
    netG = Generator(config.ngpu).to(config.device)

    # Handle multi-gpu if desired
    if (config.device.type == 'cuda') and (config.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(config.ngpu)))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netG.apply(Weights_Init)

    # 2.Create the Discriminator
    netD = Discriminator(config.ngpu).to(config.device)

    # Handle multi-gpu if desired
    if (config.device.type == 'cuda') and (config.ngpu > 1):
        netD = nn.DataParallel(netD, list(range(config.ngpu)))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netD.apply(Weights_Init)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(64, config.nz, 1, 1, device=config.device)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config.lr, betas=(config.beta1, 0.999))

    # Training Loop
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []

    start = time.time()
    for epoch in range(agrs.epochs):
        print(f"Epoch {epoch+1} of {agrs.epochs}")
        errD, errG, D_x, D_G_z1, D_G_z2 = train(netD, netG, train_loader, train_dataset, optimizerD, optimizerG, criterion)
        D_losses.append(errD)
        G_losses.append(errG)

        print(f"Generator Loss: {errD:.4f}, Discriminator Loss: {errG:.4f}, D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    end = time.time()
    print(f"Training time: {(end-start)/60:.3f} minutes")
    VisualLoss(G_losses, D_losses)

    checkpoint = {"state_dict": netG.state_dict(), "optimizer": optimizerG.state_dict()}
    Save_Checkpoint(checkpoint, filename=config.checkpoint_file)
    RealvsFake(train_loader, img_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default=config.dataroot)
    parser.add_argument('--epochs', type=int, default=config.num_epochs)
    args = parser.parse_args()
    main(args)