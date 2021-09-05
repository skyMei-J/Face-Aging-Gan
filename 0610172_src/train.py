
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
#import torch.distributed as dist
from torch import distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm

from model import Generator, Discriminator
from dataset import thumbnail128


def reduce_loss_dict(loss_dict):
    world_size = get_world_size()

    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if dist.get_rank() == 0:
            losses /= world_size

        reduced_losses = {k: v for k, v in zip(keys, losses)}

    return reduced_losses

def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()

def reduce_sum(tensor):
    if not dist.is_available():
        return tensor

    if not dist.is_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths







def train(loader, generator, discriminator, g_optim, d_optim, device, ITERATION, SAVING_POINT):
    loader = sample_data(loader)

    pbar = range(ITERATION)

    pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}


    for idx in pbar:
        if idx < ITERATION:
            i = idx 

            real_img = next(loader)
            real_img = real_img.to(device)

            for p in generator.parameters():
                p.requires_grad = False
            for p in discriminator.parameters():
                p.requires_grad = True

            noise = mixing_noise(BATCH, LATENT_SIZE, 0.9, device)
            fake_img, _ = generator(noise)
            fake_pred = discriminator(fake_img)

            real_pred = discriminator(real_img)
            d_loss = d_logistic_loss(real_pred, fake_pred)

            loss_dict['d'] = d_loss
            loss_dict['real_score'] = real_pred.mean()
            loss_dict['fake_score'] = fake_pred.mean()

            discriminator.zero_grad()
            d_loss.backward()
            d_optim.step()

            d_regularize = i % 16 == 0

            if i % 16 == 0:
                real_img.requires_grad = True
                real_pred = discriminator(real_img)
                r1_loss = d_r1_loss(real_pred, real_img)

                discriminator.zero_grad()
                (10 / 2 * r1_loss * 16 + 0 * real_pred[0]).backward()

                d_optim.step()

            loss_dict['r1'] = r1_loss

            for p in generator.parameters():
                p.requires_grad = True

            for p in discriminator.parameters():
                p.requires_grad = False

            
            noise = mixing_noise(BATCH, LATENT_SIZE, 0.9, device)
            fake_img, _ = generator(noise)
            fake_pred = discriminator(fake_img)
            g_loss = g_nonsaturating_loss(fake_pred)

            loss_dict['g'] = g_loss

            generator.zero_grad()
            g_loss.backward()
            g_optim.step()


            if i % 4 == 0:
                path_batch_size = max(1, int(BATCH /2))
                noise = mixing_noise(path_batch_size, LATENT_SIZE, 0.9, device)
                
                fake_img, latents = generator(noise,return_latents = True)

                path_loss, mean_path_length, path_lengths = g_path_regularize(fake_img, latents, 0)

                generator.zero_grad()
                weighted_path_loss = 8 * path_loss


                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

                weighted_path_loss.backward()

                g_optim.step()


            loss_reduced = reduce_loss_dict(loss_dict)

            d_loss_val = loss_reduced['d'].mean().item()
            g_loss_val = loss_reduced['g'].mean().item()
            print('d:',d_loss_val,' g:',g_loss_val)



            if (i) % SAVING_POINT == 0:
                torch.save(
                    {
                        'g': generator.state_dict(),
                        'd': discriminator.state_dict(),
                        'g_optim': g_optim.state_dict(),
                        'd_optim': d_optim.state_dict(),
                    },
                    f'CHECKPOINT/ckpt_{str(i).zfill(6)}.pt',
                )
            torch.cuda.empty_cache()
        else:
            print('finish_training')
            break
        


if __name__ == '__main__':
    
    device = 'cuda'
    label_PATH = 'train_label.txt'
    load_data_path='/data/DATALOADER/train/'
    SAVING_POINT = 10000
    BATCH = 4
    SIZE = 128 # thumbnail 128x128
    ITERATION = 200000
    LATENT_SIZE = 512
    CHECKPOINT=None#'CHECKPOINT/ckpt_000000.pt'

    if not os.path.exists('CHECKPOINT/'):
        os.makedirs('CHECKPOINT/')
    

    G = Generator(128,512,8,channel_multiplier=2).to(device)
    D = Discriminator(128, channel_multiplier=2).to(device)
    
    #generator
    generator_op    = optim.Adam(G.parameters(),    lr=0.0016, betas=(0, 0.9919))
    
    #discriminator
    discriminator_op= optim.Adam(D.parameters(),lr=0.005647, betas=(0, 0.9905))
    

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = thumbnail128(root_dir=load_data_path,img_path = label_PATH, transform=transform)
    
    
    loader = data.DataLoader(
        trainset,
        batch_size=BATCH,
        sampler=data.RandomSampler(trainset),
        num_workers=4,
        drop_last=True
    )


    train(loader, G, D, generator_op, discriminator_op, device, ITERATION,SAVING_POINT)
