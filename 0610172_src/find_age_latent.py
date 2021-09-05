import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import lpips
from model import Generator
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn

def noise_regularize(noises):
    loss = 0
    
    for noise in noises:
        shape_2 = noise.shape[2]
        while True:
            tmp1 = torch.roll(noise, shifts=1, dims=3)
            tmp1 = tmp1.mean().pow(2)
            tmp2 = torch.roll(noise, shifts=1, dims=2)
            test = torch.roll(noise, shifts=1, dims=3)
            tmp2 = tmp2.mean().pow(2)
            test = tmp2.pow(2)
            loss = tmp1+tmp2+loss
            ans = []
            for jj in range(10):
                ans.append(loss)
                tmp = shape_2/2
#             print(ans,tmp)
            tmp3 = [1, 1, int(shape_2/2), 2, int(shape_2/2), 2]

            if shape_2 <= 8:
                return loss
            noise = noise.reshape(tmp3)
            noise = noise.mean([3,5])
            shape_2 = int(shape_2/2)

    return loss



def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


    
if __name__ == '__main__':
    device = 'cuda'
    ckpt = 'checkpoint/110000.pt'
    train_img_path = '/data/DATA/'
    learning_rate = 0.1
    step=300
    noise_num = 0.05
    noise_ramp = 0.75
    noise_regularize_num = 1e5
    mean_square_error = 0.1
    batch_size = 1
    n_mean_latent = 10000
    REPRESENTING_LATENT_NUM = 50

    resize = min(128, 256)
    
        
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    g_ema = Generator(128, 512, 8)
    g_ema.load_state_dict(torch.load(ckpt)['g_ema'], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)
    

    dirs = os.listdir(  train_img_path )
    dirs.sort()
    for jj, file in enumerate(dirs):
        imglist = os.listdir( train_img_path+file+'/'+file)
        imglist.sort()
        for kk,one_img in enumerate(imglist):
            print(file)
            if (kk<=REPRESENTING_LATENT_NUM): #pick first 10 latent to present the age 'file'

                imgs = []
                print('=====Now processing:', train_img_path+file+'/'+file,'/',one_img,'(#',kk,')================')
                if(one_img[-3:]=='png'):

                    img = transform(Image.open(train_img_path+file+'/'+file+'/'+one_img).convert('RGB'))
                    imgs.append(img)
                    imgs = torch.stack(imgs, 0).to(device)

                    with torch.no_grad():
                        noise_sample = torch.randn(n_mean_latent, 512, device=device)
                        latent_out = g_ema.style(noise_sample)

                        latent_mean = latent_out.mean(0)
                        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

                    percept = lpips.PerceptualLoss(
                        model='net-lin', net='vgg', use_gpu=device.startswith('cuda')
                    )

                    noises = g_ema.make_noise()

                    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(batch_size, 1)

                    latent_in.requires_grad = True

                    for noise in noises:
                        noise.requires_grad = True

                    optimizer = optim.Adam([latent_in] + noises, lr=learning_rate)

                    pbar = tqdm(range(step))


                    for i in pbar:
                        t = i / step
                        lr = get_lr(t, learning_rate)
                        optimizer.param_groups[0]['lr'] = lr
                        noise_strength = latent_std * noise_num * max(0, 1 - t / noise_ramp) ** 2

                        latent_n = latent_in + torch.randn_like(latent_in) * noise_strength.item()

                        img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

                        batch, channel, height, width = img_gen.shape


                        p_loss = percept(img_gen, imgs).sum()
                        n_loss = noise_regularize(noises)
                        L1_loss = F.l1_loss(img_gen, imgs)

                        loss = p_loss + noise_regularize_num * n_loss + mean_square_error * L1_loss
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        for noise in noises:
                            mean = noise.mean()
                            std = noise.std()

                            noise.data.add_(-mean).div_(std)


                        pbar.set_description(
                            (
                                f'perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};'
                                f' L1_loss: {L1_loss.item():.4f}; lr: {lr:.4f}'
                            )
                        )


                    result_file = {'noises': noises}

                    print(latent_in.detach().clone().shape)
                    filename =  train_img_path+file+'/'+str(one_img)+'.pt'
                    torch.save(latent_in.detach().clone(),filename)

                    print('save at:',filename+' done')


                    del noise
                    del latent_in
                    torch.cuda.empty_cache()
