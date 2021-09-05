import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd

import lpips
from model import Generator

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





def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to('cpu')
        .numpy()
    )
# if __name__ == '__main__':
#     torch.cuda.empty_cache()
#     device = 'cuda'
#     learning_rate = 0.1
#     step=100
#     noise_num = 0.05
#     noise_ramp = 0.75
#     noise_regularize_num = 1e5
#     mean_square_error = 0
#     aging_answer_path = '../imgggg/'
#     print("DATASET:",DATASET)
#     print("CHECKPOINT:",CHECKPOINT)
#     print("aging_answer_path:",aging_answer_path)
#     print("lr:",learning_rate)
#     print("step",step)
#     n_mean_latent = 10000
#     resize = min(128, 256)
#     transform = transforms.Compose(
#         [
#             transforms.Resize(resize),
#             transforms.CenterCrop(resize),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
#         ]
#     )
    
#     g_ema = Generator(128, 512, 8)
#     g_ema.load_state_dict(torch.load('checkpoint/110000.pt')['g_ema'], strict=False)
#     g_ema.eval()
#     g_ema = g_ema.to(device)
    
            
#     test = pd.read_csv('../test_label.txt',header=None,sep="\t",names = ['name','age'])
#     test_labels = test ['age']
#     test_names = test['name']
#     test_labels = test_labels.to_list()
#     test_names =  test_names.to_list()

#     desired = pd.read_csv('../test_desired_age.txt',header=None,sep="\t",names = ['name','age'])
#     desired_labels = desired['age']
#     desired_labels = desired_labels.to_list()
    
#     for count_test, test_name in enumerate(test_names):
#         test_path = '/data/thumbnails128x128/'+str(test_name[0:2])+'000/'+test_name
#         if ( (test_path[-3:]=='png' )and (int(test_name[0:5])>=65000 )) :
            
#             imgs = []
#             img = transform(Image.open(test_path).convert('RGB'))
#             imgs.append(img)

#             imgs = torch.stack(imgs, 0).to(device)



#             with torch.no_grad():
#                 noise_sample = torch.randn(n_mean_latent, 512, device=device)
#                 latent_out = g_ema.style(noise_sample)

#                 latent_mean = latent_out.mean(0)
#                 latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

#             percept = lpips.PerceptualLoss(
#                 model='net-lin', net='vgg', use_gpu=device.startswith('cuda')
#             )

#             noises = g_ema.make_noise()

#             latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(2, 1)


#             latent_in.requires_grad = True

#             for noise in noises:
#                 noise.requires_grad = True

#             optimizer = optim.Adam([latent_in] + noises, lr=learning_rate)

#             pbar = tqdm(range(step))
#             latent_path = []

#             for i in pbar:
#                 t = i / step
#                 lr = get_lr(t, learning_rate)
#                 optimizer.param_groups[0]['lr'] = learning_rate
#                 noise_strength = latent_std * noise_num * max(0, 1 - t / noise_ramp) ** 2
#                 latent_n = latent_noise(latent_in, noise_strength.item())

#                 img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

#                 batch, channel, height, width = img_gen.shape

#                 if height > 256:
#                     factor = height // 256

#                     img_gen = img_gen.reshape(
#                         batch, channel, height // factor, factor, width // factor, factor
#                     )
#                     img_gen = img_gen.mean([3, 5])

#                 p_loss = percept(img_gen, imgs).sum()
#                 n_loss = noise_regularize(noises)
#                 mse_loss = F.mse_loss(img_gen, imgs)

#                 loss = p_loss + noise_regularize_num * n_loss + mean_square_error * mse_loss

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 noise_normalize_(noises)

#                 pbar.set_description(
#                     (
#                         f'perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};'
#                         f' mse: {mse_loss.item():.4f}; lr: {lr:.4f}'
#                     )
#                 )
#             latent_path.append(latent_in.detach().clone())
#             result_file = {'noises': noises}

#             #aging

#             latent_original = torch.load('average/'+str(test_labels[count_test])+'_average.pt')
#             latent_desired = torch.load('average/'+str(desired_labels[count_test])+'_average.pt')
#             aging_latent = latent_path[-1]-latent_original+latent_desired

#             img_gen, LATENT = g_ema([aging_latent], input_is_latent=True, noise=noises, return_latents=True)
#             img_ar = make_image(img_gen)

#             result_file[test_path] = {'img': img_gen[0], 'latent': latent_in[0]}
#             img_name = test_name[0:5] + '_aged.png'
#             pil_img = Image.fromarray(img_ar[0])

            
#             if not os.path.exists(aging_answer_path):
#                 os.makedirs(aging_answer_path)
#             pil_img.save(aging_answer_path+img_name)


#             #reconstruction
#             recon_latent = aging_latent-latent_desired+latent_original
#             img_gen, LATENT = g_ema([recon_latent], input_is_latent=True, noise=noises, return_latents=True)
#             img_ar = make_image(img_gen)

#             result_file[test_path] = {'img': img_gen[0], 'latent': latent_in[0]}
#             img_name = test_name[0:5] + '_rec.png'
#             pil_img = Image.fromarray(img_ar[0])

#             pil_img.save(aging_answer_path+img_name)


#             del noise
#             del latent_in
#             torch.cuda.empty_cache()
if __name__ == '__main__':
    
    print('setting up parameter...')
    DATASET = '/data/thumbnails128x128/'
    CHECKPOINT = "checkpoint/110000.pt"
    aging_answer_path = '../hahahahaha/'
    learning_rate = 0.1
    sumloss = []
    step=100

    print("DATASET:",DATASET)
    print("CHECKPOINT:",CHECKPOINT)
    print("aging_answer_path:",aging_answer_path)
    print("lr:",learning_rate)
    print("step",step)

    transform = transforms.Compose(
        [
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    
        ]
    )
    
    GAN = Generator(128, 512, 8)
    GAN.load_state_dict(torch.load(CHECKPOINT)['g'], strict=False)
    GAN.eval()
    GAN = GAN.to('cuda')
    
            
    test = pd.read_csv('test_label.txt',header=None,sep="\t",names = ['name','age'])
    test_labels = test ['age']
    test_names = test['name']
    test_labels = test_labels.to_list()
    test_names =  test_names.to_list()

    desired = pd.read_csv('test_desired_age.txt',header=None,sep="\t",names = ['name','age'])
    desired_labels = desired['age']
    desired_labels = desired_labels.to_list()
    
    for count_test, test_name in enumerate(test_names):
        test_path = DATASET +str(test_name[0:2])+'000/'+test_name
        if(test_path[-3:]=='png'):
            imgs = []
            img = transform(Image.open(test_path).convert('RGB'))
            imgs.append(img)

            imgs = torch.stack(imgs, 0).to('cuda')


            with torch.no_grad():
                noise_sample = torch.randn(10000, 512, device='cuda')
                latent_out = GAN.style(noise_sample)
                latent_mean = 0
                latent_mean = latent_out.mean(0)
                latent_std = 0
                latent_std = ((latent_out - latent_mean).pow(2).sum() / 10000) ** 0.5
            percept = lpips.PerceptualLoss()
            latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(2, 1)

            latent_in.requires_grad = True
            
            
            noises = GAN.make_noise()


            optimizer = optim.Adam([latent_in] + noises, lr=learning_rate)

            latent_path = []

            for i in range(step):
                t = i / step

                optimizer.param_groups[0]['lr'] = learning_rate
                noise_strength = latent_std * 0.05 * max(0,1-t/0.75)**2
                latent_n = torch.randn_like(latent_in)*noise_strength.item() + latent_in

                img_gen, _ = GAN([latent_n], input_is_latent=True, noise=noises)

                batch, channel, height, width = img_gen.shape

                # N
                p_loss = percept(img_gen, imgs).sum()#Nx3xHxW
                B_loss = p_loss
                n_loss = noise_regularize(noises)
                mse_loss = F.mse_loss(img_gen, imgs)
                loss_fn = torch.nn.MSELoss(reduction='sum')
                l1_loss = F.l1_loss(img_gen, imgs)
                loss = 0.00001*n_loss + 0.0000*mse_loss + 0.0000*l1_loss+p_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for noise in noises:
                    noise.data.add_(-noise.mean()).div_(noise.std())

                print(f' loss: {mse_loss.item():.4f};',i,'/',step)
            latent_path.append(latent_in.detach().clone())
            result_file = {'noises': noises}

            #aging

            latent_original = torch.load('average/'+str(test_labels[count_test])+'_average.pt')
            latent_desired = torch.load('average/'+str(desired_labels[count_test])+'_average.pt')
            checklist = []
            aging_latent = latent_path[-1]-latent_original+latent_desired

            generated_image, _ = GAN([aging_latent], input_is_latent=True, noise=noises)
            generated_image = generated_image.detach()
            checklist.append(generated_image)
            generated_image = generated_image.clamp_(min=-1, max=1)
            generated_image = generated_image.add(1).div_(2).mul(255).type(torch.uint8)
            generated_image = generated_image.permute(0, 2, 3, 1)
            generated_image = generated_image.to('cpu').numpy()
            print("aging")
            
#             img_ar = make_image(generated_image)

            result_file[test_path] = {'img': generated_image[0], 'latent': latent_in[0]}
            img_name = test_name[0:5] + '_aged.png'
            pil_img = Image.fromarray(generated_image[0])
            
            
            if not os.path.exists(aging_answer_path):
                os.makedirs(aging_answer_path)
            pil_img.save(aging_answer_path+img_name)
            print("Aging: ",aging_answer_path+img_name,' Done')

            #reconstruction
            checklist = []
            recon_latent = aging_latent-latent_desired+latent_original
            checklist.append(recon_latent)
            generated_image, _ = GAN([recon_latent], input_is_latent=True, noise=noises)
            checklist.append(generated_image)
            generated_image = generated_image.detach()
            generated_image = generated_image.clamp_(min=-1, max=1)
            generated_image = generated_image.add(1).div_(2).mul(255).type(torch.uint8)
            generated_image = generated_image.permute(0, 2, 3, 1)
            generated_image = generated_image.to('cpu').numpy()

            result_file[test_path] = {'img': generated_image[0], 'latent': latent_in[0]}
            img_name = test_name[0:5] + '_rec.png'
            pil_img = Image.fromarray(generated_image[0])

            pil_img.save(aging_answer_path+img_name)
            print("Reconstruct: ",aging_answer_path+img_name,' Done')
            
            del latent_in
            torch.cuda.empty_cache()

