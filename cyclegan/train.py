import torch
from dataset import KvairVCEDataset
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator


def train_fn(disc_K,disc_V, gen_K, gen_V, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler):
    #create a loop
    loop = tqdm(loader, leave= True)

    for idx, (Kvair, VCE) in enumerate(loop):
        Kvair = Kvair.to(config.DEVICE)
        VCE = VCE.to(config.DEVICE)

        # train Discriminator H and Z
        with torch.cuda.amp.autocast():
            fake_VCE = gen_V(Kvair)
            D_V_real = disc_V(VCE) # use a real vce picture
            D_V_fake = disc_V(fake_VCE.detach()) # use a fake vce picture
            D_V_real_loss = mse(D_V_real, torch.ones_like(D_V_real))
            D_V_fake_loss = mse(D_V_fake, torch.ones_like(D_V_fake))
            D_V_loss = D_V_real_loss + D_V_fake_loss

            fake_Kvair = gen_K(VCE)
            D_K_real = disc_K(Kvair) # use a real vce picture
            D_K_fake = disc_K(fake_Kvair.detach()) # use a fake vce picture
            D_K_real_loss = mse(D_K_real, torch.ones_like(D_V_real))
            D_K_fake_loss = mse(D_K_fake, torch.ones_like(D_V_fake))
            D_K_loss = D_K_real_loss + D_K_fake_loss

            D_loss = (D_K_loss + D_V_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with torch.cuda.amp.autocast():
            # adversirial loss for both generatots
            D_V_fake = disc_V(fake_Kvair)
            D_K_fake = disc_K(fake_VCE)
            loss_G_V = mse(D_V_fake, torch.ones_like(D_V_fake))
            loss_G_K = mse(D_K_fake, torch.ones_like(D_K_fake))

            # cycle loss
            # get back to the original one
            cycle_K = gen_K(fake_VCE)
            cycle_V = gen_V(fake_Kvair)
            cycle_loss_V = L1(VCE, cycle_V)
            cycle_loss_K = L1(Kvair, cycle_K)

            #identity loss
            identity_K = gen_K(Kvair)
            identity_V = gen_V(VCE)
            identity_loss_K = L1(Kvair, identity_K)
            identity_loss_V = L1(VCE, identity_V)

            G_loss = (loss_G_V+loss_G_K+cycle_loss_K+cycle_loss_V+identity_loss_K+identity_loss_V)
        
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        #train 
        if idx % 200 == 0:
            save_image(fake_Kvair*0.5+0.5, f"saved_images/Kvair_{idx}.png")
            save_image(fake_VCE*0.5+0.5, f"saved_images/VCE_{idx}.png")

    pass

def main():
    disc_K = Discriminator(in_channels=3).to(config.DEVICE)
    disc_V = Discriminator(in_channels=3).to(config.DEVICE)
    gen_K = Generator(img_channels = 3, num_residuals=9).to(config.DEVICE)
    gen_V = Generator(img_channels = 3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_K.parameters()) + list(disc_V.parameters()),
        lr = config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    opt_gen = optim.Adam(
        list(gen_K.parameters()) + list(gen_V.parameters()),
        lr = config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_K, gen_K, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_V, gen_V, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_K, disc_K, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_V, disc_V, opt_disc, config.LEARNING_RATE,
        )
    
    dataset = KvairVCEDataset(root_Kvair="dataset/train/Kvair", root_VCE="dataset/train/VCE", transform=config.transforms)
    #dataset is for the tainning

    loader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    #in each epochs, save the check point

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_K,disc_V, gen_K, gen_V, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(gen_K, opt_gen, filename=config.CHECKPOINT_GEN_K)
            save_checkpoint(gen_V, opt_gen, filename=config.CHECKPOINT_GEN_V)
            save_checkpoint(disc_K, opt_disc, filename=config.CHECKPOINT_CRITIC_K)
            save_checkpoint(disc_V, opt_disc, filename=config.CHECKPOINT_CRITIC_V)

main()