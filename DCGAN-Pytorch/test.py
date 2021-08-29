import argparse
import torch
import config
from model import Generator
import torchvision.utils as vutils
from utils import Imshow, Load_Checkpoint

def predict(netG):
    netG.eval()
    img_list = []

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(64, config.nz, 1, 1, device=config.device)

    for i in range(64): # created 64 fake image
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
    Imshow(img_list)

def main(args):
    model = Generator(config.ngpu).to(config.device)
    NetG =  Load_Checkpoint(torch.load(args.weight), model)
    
    predict(NetG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default=config.checkpoint_file)
    args = parser.parse_args()
    main(args)