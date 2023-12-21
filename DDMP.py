
import random
import imageio
import numpy as np
from argparse import ArgumentParser
from typing import Callable, Iterable
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import einops
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from uniplot import plot
from scipy.stats import describe
from pandas import DataFrame
from rich.table import Table
from rich.console import Console

from torchvision.transforms import Compose, ToTensor, Lambda

# Setting reproducibility

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

transform = Compose([
    ToTensor(),
    Lambda(lambda x: (x - 0.5) * 2)]
)


 
def show_images(images : Iterable, title : str = ""):
    """
    show_images show images from images list 

    Parameters
    ----------
    images : Iterable
        _description_
    title : str, optional
        _description_, by default ""
    """
    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()  
    
    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0])
                idx += 1
    fig.suptitle(title, fontsize=20)

    # Showing the figure
    plt.show()

 
def show_first_batch(loader):
    for batch in loader:
        print(batch.shape)
        show_images(batch, "Images in the first batch")
        break

 
# DDPM class
class MyDDPM(nn.Module):
    
    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None, image_chw = (1, 64, 64)):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(
            device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0 : Iterable, t : int, eta=None):
        
        """
        forward Make input image more noisy (we can directly skip to the desired step)_

        Parameters
        ----------
        x0 : Iterable
            Original image 
        t : int
            wanted timestep 
        eta : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """        
        # M
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy

    def backward(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.network.forward(x, t)

 
def show_forward(ddpm, loader, device):
    # Showing the forward process
    for batch in loader:
        imgs = batch

        show_images(imgs, "Original images")

        for percent in [0.25, 0.5, 0.75, 1]:
            show_images(
                ddpm(imgs.to(device
                             
                             ),
                     [int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))]),
                f"DDPM Noisy images {int(percent * 100)}%"
            )
            
        break

 
def generate_new_images(ddpm : MyDDPM, n_samples : int = 16, device=None, frames_per_gif : int = 10, 
                        gif_name="sampling.gif", c : int = 1, h : int = 64, w : int = 64) -> Iterable :
    """
    generate_new_images Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples

    Parameters
    ----------
    ddpm : MyDDPM
        DDPM  model
    n_samples : int, optional
        number of samples to be generated, by default 16
    device : _type_, optional
        _description_, by default None
    frames_per_gif : int, optional
        _description_, by default 100
    gif_name : str, optional
        _description_, by default "sampling.gif"
    c : int, optional
        color channel, by default 1
    h : int, optional
        image's height, by default 64
    w : int, optional
        images's width, by default 64

    Returns
    -------
    Iterable
        array of newly generated images
    """    

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # Starting from random noise
        x = torch.randn(n_samples, c, h, w).to(device)
        for idx, t in enumerate(tqdm(list(range(ddpm.n_steps))[::-1], desc = "Generating new images", colour="#A3BE8C")):
                    # Estimating noise to be removed
                    time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
                    eta_theta = ddpm.backward(x, time_tensor)

                    alpha_t = ddpm.alphas[t]
                    alpha_t_bar = ddpm.alpha_bars[t]

                    # Partially denoising the image
                    x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

                    if t > 0:
                        z = torch.randn(n_samples, c, h, w).to(device)

                        beta_t = ddpm.betas[t]
                        sigma_t = beta_t.sqrt()
                        
                        x = x + sigma_t * z
                

    return x

 
class MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize
    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out

 
def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding

 
class MyUNet(nn.Module):
    def __init__(self, n_steps = 1000, time_emb_dim = 100):
        super(MyUNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock((1, 64, 64), 1, 10),
            MyBlock((10, 64, 64), 10, 10),
            MyBlock((10, 64, 64), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            MyBlock((10, 32, 32), 10, 20),
            MyBlock((20, 32, 32), 20, 20),
            MyBlock((20, 32, 32), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            MyBlock((20, 16, 16), 20, 40),
            MyBlock((40, 16, 16), 40, 40),
            MyBlock((40, 16, 16), 40, 40)
        )
       
        self.down3 = nn.Conv2d(40, 40, 4, 2, 1)
      
        self.te4 = self._make_te(time_emb_dim, 40)
        self.b4 = nn.Sequential(
            MyBlock((40, 8, 8), 40, 80),
            MyBlock((80, 8, 8), 80, 80),
            MyBlock((80, 8, 8), 80, 80)
        )
        self.down4 = nn.Conv2d(80, 80, 4, 2, 1)
        
        self.te5 = self._make_te(time_emb_dim, 80)
        self.b5 = nn.Sequential(
            MyBlock((80, 4, 4), 80, 160),
            MyBlock((160, 4, 4), 160, 160),
            MyBlock((160, 4, 4), 160, 160)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(160, 160, 4, 1, 0),
            nn.SiLU())

        
        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 160)
        self.b_mid = nn.Sequential(
            MyBlock((160, 1, 1), 160, 80),
            MyBlock((80, 1, 1), 80, 80),
            MyBlock((80, 1, 1), 80, 160)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(160, 160, 4, 1, 0),
            nn.SiLU()
        )

        self.te6 = self._make_te(time_emb_dim, 320)
        self.b6 = nn.Sequential(
            MyBlock((320, 4, 4), 320, 160),
            MyBlock((160, 4, 4), 160, 80),
            MyBlock((80, 4, 4), 80, 80)
        )

        self.up2 = nn.ConvTranspose2d(80, 80, 4, 2, 1)
        self.te7 = self._make_te(time_emb_dim, 160)
        self.b7 = nn.Sequential(
            MyBlock((160, 8, 8), 160, 80),
            MyBlock((80, 8, 8), 80, 40),
            MyBlock((40, 8, 8), 40, 40)
        )
        
        self.up3 = nn.ConvTranspose2d(40, 40, 4, 2, 1)
        self.te8 = self._make_te(time_emb_dim, 80)
        self.b8 = nn.Sequential(
            MyBlock((80, 16, 16), 80, 40),
            MyBlock((40, 16, 16), 40, 20),
            MyBlock((20, 16, 16), 20, 20)
        )
        
        self.up4 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te9 = self._make_te(time_emb_dim, 40)
        self.b9 = nn.Sequential(
            MyBlock((40, 32, 32), 40, 20),
            MyBlock((20, 32, 32), 20, 10),
            MyBlock((10, 32, 32), 10, 10)
        )

        
        self.up7 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            MyBlock((20, 64, 64), 20, 10),
            MyBlock((10, 64, 64), 10, 10),
            MyBlock((10, 64, 64), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

    def forward(self, x, t):
        # x is (N, 2, 64, 64) (image with positional embedding stacked on channel dimension)
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 10, 64, 64)
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (N, 20, 32, 32)
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (N, 40, 16, 16)
        out4 = self.b4(self.down3(out3) + self.te4(t).reshape(n, -1, 1, 1))
        out5 = self.b5(self.down4(out4) + self.te5(t).reshape(n, -1, 1, 1))
        
        out_mid = self.b_mid(self.down5(out5) + self.te_mid(t).reshape(n, -1, 1, 1)) 
        
        out6 = torch.cat((out5, self.up1(out_mid)), dim=1)  
        out6 = self.b6(out6 + self.te6(t).reshape(n, -1, 1, 1))  
        
        out7 = torch.cat((out4, self.up2(out6)), dim=1)  
        out7 = self.b7(out7 + self.te7(t).reshape(n, -1, 1, 1))  

        out8 = torch.cat((out3, self.up3(out7)), dim=1) 
        out8 = self.b8(out8 + self.te8(t).reshape(n, -1, 1, 1))  # (N, 10, 32, 32)
        
        out9 = torch.cat((out2, self.up4(out8)), dim=1)  
        out9 = self.b9(out9 + self.te9(t).reshape(n, -1, 1, 1))

        out = torch.cat((out1, self.up7(out9)), dim=1) 
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 1, 64, 64)
        
        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )
        
def training_loop(ddpm : MyDDPM, loader : DataLoader, n_epochs : int, optim , device, display = False, store_path="ddpm_model.pt"):
    """
    training_loop train the DDPM model using loader

    Parameters
    ----------
    ddpm : MyDDPM
        _description_
    loader : DataLoader
        datasets loader
    n_epochs : int
        number of epochs
    optim : _type_
        _description_
    device : _type_
        _description_
    display : bool, optional
        display image at each epoch, by default False
    store_path : str, optional
        _description_, by default "ddpm_model.pt"
    """    
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps
    loss_history = []
    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#EBCB8B"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", postfix = {'loss' : epoch_loss}, colour="#BF616A")):
            # Loading data
            x0 = batch.to(device)
            n = len(x0)
            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)
            # Computing the noisy image based on x0 and the time-step (forward process)
            noisy_imgs = ddpm(x0, t, eta)

            # Getting model estimation of noise based on the images and the time-step
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_theta, eta)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)
        # Display images generated at this epoch
        console = Console()
        infos = DataFrame(describe(np.array(generate_new_images(ddpm, 1, device=device)), axis = None))
        infos.insert(0, 'Stat', ['Nobs', 'Min-Max', 'Mean', 'Variance', 'Skewness', 'Kurtosis'])   
        table = Table('Dataset Processed Infos')
        table.add_row(infos.to_string(float_format=lambda _: '{:.4f}'.format(_)))
        console.print(table)
        
        if display:
            show_images(generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            loss_history.append(best_loss)
            torch.save(ddpm.state_dict(), store_path)
            
    np.save("loss_history.npy", np.array(loss_history))