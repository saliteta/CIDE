import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Critic, Generator, initialize_weight
from utils import gradient_penalty

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNEL_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 1
FEATURES_DISC = 64
FEATURE_GEN = 64
CRITIC_ITERATION = 5
WEIGHT_CLIP = 0.01
LAMBDA_GP = 10

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(CHANNEL_IMG)], [0.5 for _ in range(CHANNEL_IMG)]),
        ]
)

dataset = datasets.ImageFolder(root="celeb_dataset", transform=transform)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
gen = Generator(Z_DIM, CHANNEL_IMG, FEATURE_GEN).to(DEVICE)
critic = Critic(CHANNEL_IMG, FEATURES_DISC).to(DEVICE)
initialize_weight(gen)
initialize_weight(critic)


opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))# beta is from the papaer
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(DEVICE)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(DEVICE)


        for _ in range(CRITIC_ITERATION):
            noise = torch.randn((BATCH_SIZE, Z_DIM,1 ,1)).to(DEVICE)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic,real, fake, device=DEVICE)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))+LAMBDA_GP*gp
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

            
        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1