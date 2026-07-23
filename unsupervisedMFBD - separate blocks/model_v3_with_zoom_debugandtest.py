import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.init as init
import util
import zern
import kl_modes
from einops import rearrange, repeat
import torch.nn.functional as F
# Import your image reading library (e.g., fits from astropy, imageio, or numpy)
#from astropy.io import fits 
import tifffile as tiff
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use('Agg')  # Headless backend for HPC clusters (no display needed)
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import random

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, bn=True, activation=True):
        """Convolutional block : BN+RELU+CONV
        The CONV uses reflection padding
        BN and RELU can be on/off depending on the keywords "bn" and "activation"
        
        Args:
            inplanes (int): number of input channels
            outplanes (int): number of output channels
            kernel_size (int, optional): Kernel size. Defaults to 3.
            stride (int, optional): Stride. Defaults to 1.
            bn (bool, optional): Use batch normalization. Defaults to True.
            activation (bool, optional): Use activation. Defaults to True.
        """
        super(ConvBlock, self).__init__()

        self.use_bn = bn
        self.use_activation = activation

        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride)
        self.reflection = nn.ReflectionPad2d(int((kernel_size-1)/2))

        if (bn):
            self.bn = nn.BatchNorm2d(inplanes)

        self.elu = nn.ELU(inplace=False)

    def forward(self, x):
        if (self.use_bn):
            out = self.bn(x)
            out = self.elu(out)
            out = self.reflection(out)
            out = self.conv(out)

        else:
            out = self.reflection(x)
            out = self.conv(out)
            if (self.use_activation):
                out = self.elu(out)

        return out
    
class CNN(nn.Module):
    def __init__(self, n, n_lstm):
        """Neural network to estimate latent features from a set of images
        An encoder is applied in parallel to all pairs of images so that a vector
        of 128 features is obtained from each pair of images. 
        
        Args:
            n (int, optional): Number of channels in the hidden convolutional layers. Defaults to 32.
            n_lstm: number of input channels for the afterwards lstm network
        """
        super().__init__()

        self.n_lstm = n_lstm

        # self.n_modes = n_modes
        # self.npix_image = npix_image
        # self.n_frames = n_frames
        # self.device = device

        self.A01 = ConvBlock(1, n, kernel_size=9, bn=False, activation=False)

        self.C01 = ConvBlock(n, n, kernel_size=7, stride=2)
        self.C02 = ConvBlock(n, n, kernel_size=7)
        self.C03 = ConvBlock(n, n, kernel_size=7)
        self.C04 = ConvBlock(n, n, kernel_size=7)

        self.C11 = ConvBlock(n, n, kernel_size=5, stride=2)
        self.C12 = ConvBlock(n, n, kernel_size=5)
        self.C13 = ConvBlock(n, n, kernel_size=5)
        self.C14 = ConvBlock(n, n, kernel_size=5)

        self.C21 = ConvBlock(n, n, kernel_size=3, stride=2)
        self.C22 = ConvBlock(n, n, kernel_size=3)
        self.C23 = ConvBlock(n, n, kernel_size=3)
        self.C24 = ConvBlock(n, n, kernel_size=3)

        # kernel_size = 16
        # self.C41 = nn.Conv2d(n, self.n_lstm, kernel_size=kernel_size, stride=1)

        # CHANGE 1: Global adaptive pooling collapses spatial dimensions to 1x1 
        # regardless of whether the image is 128x128, 256x256, etc.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.C41 = nn.Conv2d(n, self.n_lstm, kernel_size=1, stride=1)

    def weights_init(self):
        for module in self.modules():
            kaiming_init(module)

    def forward(self, images):
        
        # # We reform the tensor from (B,Nf,1,nx,ny) to (B*Nf,1,nx,ny) so that the features
        # # are extracted for all frames of all batches in parallel
        # # B is the batch size
        # # Nf is the number of frames
        # tmp = images.view(-1, 1, self.npix_image, self.npix_image)

        # CHANGE 2: Extract spatial dimensions dynamically
        if images.dim() == 4:
            images = images.unsqueeze(2)

        B, Nf, C, H, W = images.shape
        tmp = images.view(B * Nf, C, H, W)

        # (B*Nf,2,129,128) -> (B*Nf,32,128,128)
        A01 = self.A01(tmp)

        # (B*Nf,32,128,128) -> (B*Nf,32,64,64)
        C01 = self.C01(A01)
        C02 = self.C02(C01)
        C03 = self.C03(C02)
        C04 = C01 + self.C04(C03)

        # (B*Nf,32,64,64) -> (B*Nf,32,32,32)
        C11 = self.C11(C04)
        C12 = self.C12(C11)
        C13 = self.C13(C12)
        C14 = C11 + self.C14(C13)

        # (B*Nf,32,32,32) -> (B*Nf,32,16,16)
        C21 = self.C21(C14)
        C22 = self.C22(C21)
        C23 = self.C23(C22)
        C24 = C21 + self.C24(C23)

        # # (B*Nf,32,16,16) -> (B*Nf,128,1,1)
        # out = self.C41(C24)

        # # (B*Nf,128,1) -> (B*Nf,128)
        # out = out.squeeze()

        # # (B*Nf,128) -> (B,Nf,128)
        # out = out.view(-1, self.n_frames, self.n_lstm)

        out = self.global_pool(C24)
        out = self.C41(out)

        out = torch.flatten(out, start_dim=1)
        out = out.view(B, Nf, self.n_lstm)

        return out
    
class LSTM(nn.Module):
    def __init__(self, n_modes, n_lstm):
        """Neural network to estimate the wavefront coefficients from a set of latent features.
        This one uses a recurrent architecture and works for M pairs of focused+defocused
        images. These vectors (latent features) are then fed
        into an bi-directional LSTM that provides as output another vector of size 128
        per timestep. A final linear layer projects these vectors into the modal coefficients.
        
        Args:
            latent_features: input vector with latent features
            n (int, optional): Number of channels in the hidden convolutional layers. Defaults to 32.
            n_modes (int, optional): Number of output modes. Defaults to 40.
            n_lstm: number of input channels for the lstm network
        """
        super().__init__()

        self.n_modes = n_modes
        self.n_lstm = n_lstm
        
        self.C42 = nn.Linear(2*self.n_lstm, self.n_lstm)
        self.C43 = nn.Linear(self.n_lstm, n_modes)
        
        self.elu = nn.ELU()

        self.lstm = nn.LSTM(self.n_lstm, self.n_lstm, batch_first=True, bidirectional=True, dropout=0.0)

    def weights_init(self):
        # Apply standard Kaiming initialization to all layers
        for module in self.modules():
            kaiming_init(module)

        # Override the final linear layer (C43) with near-zero initialization
        nn.init.normal_(self.C43.weight, std=1e-3)
        if self.C43.bias is not None:
            nn.init.zeros_(self.C43.bias)

    def forward(self, latent_features, lengths=None):
        """
        Args:
            latent_features (tensor): Extracted spatial features [B, Nf, n_lstm]
            lengths (tensor, optional): Actual valid frame count per batch item [B] (e.g., tensor([494, 500]))
        """
        if lengths is not None:
            # Pack sequence so LSTM ignores padded zero-frames
            packed = nn.utils.rnn.pack_padded_sequence(
                latent_features, lengths.to(torch.int64).cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, _ = self.lstm(latent_features)

        # Reshape for linear MLP projection: [B * Nf, 2 * n_lstm]
        out = out.reshape(-1, 2 * self.n_lstm)
        out = self.elu(self.C42(out))
        out = self.C43(out)
        return out

class Network(nn.Module):
    def __init__(self, device='cpu', n_modes=44, n_frames=5, pixel_size=0.042, \
        telescope_diameter=150.0, central_obscuration=0.0, wavelength=8000.0, basis_for_wavefront='zernike', npix_image=128):
        
        super().__init__()

        self.n_modes = n_modes
        self.n_frames = n_frames
        self.pixel_size = pixel_size
        self.telescope_diameter = telescope_diameter
        self.central_obscuration = central_obscuration
        self.wavelength = wavelength
        self.npix_image = npix_image
        self.basis_for_wavefront = basis_for_wavefront
        self.device = device

        print(f"Wavelength : {self.wavelength} A")
        print(f"Diameter : {self.telescope_diameter} cm")
        print(f"Central obscuration : {self.central_obscuration} cm")
        print(f"Pixel size : {self.pixel_size} arcsec")

        # Compute PSF scale, which depends on the wavelength, telescope diameter and pixel size. This is used to compute the overfill of the pupil
        # so that the final PSF has the correct pixel size
        self.overfill = util.psf_scale(self.wavelength, self.telescope_diameter, self.pixel_size)                
        if (self.overfill < 1.0):
            raise Exception(f"The pixel size is not small enough to model a telescope with D={self.telescope_diameter} cm")
            
        # Compute telescope aperture
        pupil = util.aperture(npix=self.npix_image, cent_obs = self.central_obscuration / self.telescope_diameter, spider=0, overfill=self.overfill)
        pupil = torch.tensor(pupil.astype('float32'))
            
        # Define the basis for the wavefront. This can be either Zernike or KL modes
        if (self.basis_for_wavefront == 'zernike'):
            print("Computing Zernike modes...")
            Z_machine = zern.ZernikeNaive(mask=[])
            x = np.linspace(-1, 1, self.npix_image)
            xx, yy = np.meshgrid(x, x)
            rho = self.overfill * np.sqrt(xx ** 2 + yy ** 2)
            theta = np.arctan2(yy, xx)
            aperture_mask = rho <= 1.0

            basis = np.zeros((self.n_modes, self.npix_image, self.npix_image))
            
            # Precompute all Zernike modes except for piston
            for j in range(self.n_modes):
                n, m = zern.zernIndex(j+2)
                Z = Z_machine.Z_nm(n, m, rho, theta, True, 'Jacobi')
                basis[j,:,:] = Z * aperture_mask

        if (self.basis_for_wavefront == 'kl'):
            print("Computing KL modes...")
            kl = kl_modes.KL()
            basis = kl.precalculate_covariance(npix_image = self.npix_image, n_modes_max = self.n_modes, first_noll = 1, overfill=self.overfill)

        zeros = torch.zeros((self.npix_image, self.npix_image, 1), dtype=torch.float32)

        # Register buffers so that they are moved to the GPU when the model is moved to the GPU
        self.register_buffer('zeros', zeros)
        self.register_buffer('pupil', pupil)
        self.register_buffer('basis', torch.tensor(basis.astype('float32')))

        # Define the neural network that will estimate the wavefront coefficients from a set of images
        self.cnn = CNN(n=16, n_lstm=256)
        self.cnn.weights_init()

        self.lstm = LSTM(n_modes=self.n_modes, n_lstm=256)
        self.lstm.weights_init()

    def update_telescope_basis(self, pixel_size, telescope_diameter, central_obscuration, wavelength, npix_image):
        """Recalculates the optical pupil mask and wavefront modal basis for a specific telescope configuration."""
        
        # Create a signature dict to check if recalculation is needed
        new_config = (pixel_size, telescope_diameter, central_obscuration, wavelength, npix_image)
        if self.current_config == new_config:
            return  # Skip expensive recalculation if configuration is identical to previous sample
        
        self.pixel_size = pixel_size
        self.telescope_diameter = telescope_diameter
        self.central_obscuration = central_obscuration
        self.wavelength = wavelength
        self.npix_image = npix_image
        self.current_config = new_config

        # Compute PSF scale & overfill
        self.overfill = util.psf_scale(self.wavelength, self.telescope_diameter, self.pixel_size)                
        if (self.overfill < 1.0):
            raise Exception(f"Pixel size {self.pixel_size} arcsec is not small enough to model D={self.telescope_diameter} cm")
            
        # Compute exact telescope aperture
        pupil = util.aperture(npix=self.npix_image, cent_obs=self.central_obscuration / self.telescope_diameter, spider=0, overfill=self.overfill)
        pupil = torch.tensor(pupil.astype('float32'), device=self.device)
            
        # Compute optical modal basis
        if (self.basis_for_wavefront == 'zernike'):
            Z_machine = zern.ZernikeNaive(mask=[])
            x = np.linspace(-1, 1, self.npix_image)
            xx, yy = np.meshgrid(x, x)
            rho = self.overfill * np.sqrt(xx ** 2 + yy ** 2)
            theta = np.arctan2(yy, xx)
            aperture_mask = rho <= 1.0

            basis = np.zeros((self.n_modes, self.npix_image, self.npix_image))
            for j in range(self.n_modes):
                n, m = zern.zernIndex(j+2)
                Z = Z_machine.Z_nm(n, m, rho, theta, True, 'Jacobi')
                basis[j,:,:] = Z * aperture_mask

        elif (self.basis_for_wavefront == 'kl'):
            kl = kl_modes.KL()
            basis = kl.precalculate_covariance(npix_image=self.npix_image, n_modes_max=self.n_modes, first_noll=1, overfill=self.overfill)

        zeros = torch.zeros((self.npix_image, self.npix_image, 1), dtype=torch.float32, device=self.device)

        # Register buffers dynamically on target device
        self.register_buffer('zeros', zeros, persistent=False)
        self.register_buffer('pupil', pupil, persistent=False)
        self.register_buffer('basis', torch.tensor(basis.astype('float32'), device=self.device), persistent=False)

    def compute_psfs(self, coeff):
        """Compute exact physical PSFs and OTFs from estimated wavefront coefficients."""
        # Compute real phase screens using current telescope basis
        wavefront = torch.einsum('ij,jkl->ikl', coeff, self.basis)

        # Compute generalized pupil function
        phase = self.pupil[None, :, :] * torch.exp(1j * wavefront)

        # Compute Fourier transform and autocorrelation
        ft = torch.fft.fft2(phase, norm="ortho")
        psf = (torch.conj(ft) * ft).real

        # Normalize PSF
        psf_norm = psf / torch.sum(psf, [-1, -2], keepdim=True)

        # Compute OTF
        otf = torch.fft.fft2(psf_norm, norm="ortho")

        return psf, otf, wavefront

    def loss_and_wiener_filter(self, im_ft, psf_ft, variance, lengths=None):
        """Compute MOMFBD loss function and the estimated deconvolved image. See Michiel van Noorts and Mats Löfdahl papers
        
        Args:
            focused_ft (tensor): FFT of the focused images
            defocused_ft (tensor): FFT of the defocused images
            psf_focused_ft (tensor): FFT of the focused PSF
            psf_defocused_ft (tensor): FFT of the defocused PSF
        
        
        """
        # zero out the PSFs/OTFs of padded frames before computing the loss:
        if lengths is not None:
            Nf = psf_ft.shape[1]
            lengths_dev = lengths.to(psf_ft.device)
            # mask shape: [B, Nf, 1, 1]
            mask = (torch.arange(Nf, device=psf_ft.device)[None, :] < lengths_dev[:, None])[:, :, None, None]
            psf_ft = psf_ft * mask  # Zero out OTF for padded frames
            im_ft = im_ft * mask

    
        # D = burst_ft
        # S = psf_ft
                
        # Compute S* x D
        S_star_D = torch.conj(psf_ft) * im_ft
        
        # Compute D* x S
        D_star_S = torch.conj(im_ft) * psf_ft
        
        # Compute modulus of S : |S|^2 = S* x S
        modulus_S = torch.conj(psf_ft) * psf_ft
        
        # Compute modulus of D : |D|^2 = D* x D
        modulus_D = torch.conj(im_ft) * im_ft
        
        # Compute modulus of the product between D^* and S summed for all frames
        sum_D_star_S = torch.sum(D_star_S, dim=1)
        modulus_D_star_S = torch.conj(sum_D_star_S) * sum_D_star_S

        # Wiener filter estimation of the image
        denominator = torch.sum(modulus_S, dim=1)
        # Q[..., 0] += 1e-10
        numerator = torch.sum(S_star_D, dim=1)

        # Loss function
        tmp = torch.sum(modulus_D, dim=1)

        loss = tmp - modulus_D_star_S / (variance[:, None, None] + denominator + 1e-10)

        # # This normalization is here because we use non-normalized FFTs, which
        # # lack a sqrt(Nx*Ny). It is squared because the loss function has
        # # squared FFTs
        # loss_mn = torch.mean(loss.real) / (self.npix_image**2)

        # -------------------------------------------------------------
        # With norm="ortho", spatial_pixels scaling is no longer required
        # -------------------------------------------------------------

        # # CHANGE: Use dynamic spatial pixel count for unnormalized FFT loss scaling
        # spatial_pixels = im_ft.shape[-2] * im_ft.shape[-1]

        if lengths is not None:
            # Sum only valid non-padded frame losses and divide by total valid frame count across batch
            total_valid_frames = torch.sum(lengths_dev)
            # loss_mn = (torch.sum(loss.real) / total_valid_frames) / spatial_pixels
            loss_mn = (torch.sum(loss.real) / total_valid_frames)
        else:
            # loss_mn = torch.mean(loss.real) / spatial_pixels
            loss_mn = torch.mean(loss.real)

        return numerator, denominator, loss_mn

    def forward(self, images, images_ft, variance, lengths=None):
        """
        Args:
            images (tensor): Input image stack [B, Nf, H, W] or [B, Nf, 1, H, W]
            images_ft (tensor): Fourier transform of images
            variance (tensor): Noise variance per batch item
            lengths (tensor, optional): Tensor containing valid sequence lengths per batch item [B]
        """
        
        # CHANGE 1: Dynamically retrieve sequence and spatial shape parameters
        B, Nf = images.shape[0], images.shape[1]
        H, W = images.shape[-2], images.shape[-1]

        # 1. Extraction of latent features per image [B, Nf, 256]
        latent_features = self.cnn(images)

        # 2. Sequence processing with variable stack support: [B * Nf, n_modes]
        coeff = self.lstm(latent_features, lengths=lengths)

        # Rearrange the coefficients from (B*Nf, N_modes) to (B, Nf, N_modes)
        tmp = rearrange(coeff, '(b f) m -> b f m', f=Nf, m=self.n_modes)

        # Calculate average tip-tilt (modes 0 & 1), masking out padded frames if lengths are supplied
        if lengths is not None:
            lengths_dev = lengths.to(images.device)
            
            # Create boolean mask: [B, Nf] (True for real frames, False for padded frames)
            mask = torch.arange(Nf, device=images.device)[None, :] < lengths_dev[:, None]
            mask_expanded = mask.unsqueeze(-1)

            # Sum valid frames only and divide by actual valid lengths
            sum_coeff = torch.sum(tmp * mask_expanded, dim=1)
            avg = sum_coeff / lengths_dev[:, None]
        else:
            avg = torch.mean(tmp, dim=1)

        # -----------------------------------------------------------------
        # OLD (IN-PLACE MUTATION CAUSING DETACHMENT):
        # # Force zero tip-tilt on average for all observed frames. This is done because
        # # the tip-tilt is degenerate with the image motion and cannot be estimated from the images. 
        # # The average tip-tilt is set to zero so that the estimated wavefronts are centered on the image.
        # avg[:, 2:] = 0.0
        # avg = repeat(avg, 'b m -> b f m', f=Nf)
        # avg = rearrange(avg, 'b f m -> (b f) m')
        # -----------------------------------------------------------------

        # NEW (SAFE MULTIPLICATION KEEPING AUTOGRAD INTENDED):
        mask_tt = torch.zeros_like(avg)
        mask_tt[:, :2] = 1.0  # Preserve Tip-Tilt (modes 0 and 1) only
        avg = avg * mask_tt

        avg = repeat(avg, 'b m -> b f m', f=Nf)
        avg = rearrange(avg, 'b f m -> (b f) m')

        # Compute PSFs dynamically for current frame grid
        coeff_corrected = coeff - avg
        psf, psf_ft, wavefront = self.compute_psfs(coeff_corrected, target_shape=(H, W))
        psf_ft = rearrange(psf_ft, '(b f) x y -> b f x y', f=Nf)

        # Compute physics-based loss and Wiener filter terms
        numerator, denominator, loss = self.loss_and_wiener_filter(images_ft, psf_ft, variance, lengths=lengths)
        
        return coeff_corrected, numerator, denominator, psf, psf_ft, loss
    
class MultiTelescopeStackDataset(Dataset):
    def __init__(self, root_dir: str | Path, n_frames: int = 10, crop_dim: int | None = None):
        self.root_path = Path(root_dir)
        self.n_frames = n_frames
        self.crop_dim = crop_dim  # Optional: set to e.g. 128 if you want to force center-cropping across all images
        self.samples = []
        
        # Iterate over subdirectories in root_dir
        for tel_dir in self.root_path.iterdir():
            if not tel_dir.is_dir():
                continue
                
            config_path = tel_dir / "config.json"
            if not config_path.exists():
                print(f"Warning: Skipping {tel_dir.name} because config.json was not found.")
                continue
                
            # Read telescope configuration (diameter, obscuration, pixel_size, wavelength)
            with open(config_path, "r", encoding="utf-8") as f:
                tel_config = json.load(f)
                
            # Collect all .tiff and .tif stacks in this folder
            tiff_files = list(tel_dir.glob("*.tiff")) + list(tel_dir.glob("*.tif"))
            
            for tiff_path in tiff_files:
                self.samples.append({
                    "path": tiff_path,
                    "config": tel_config
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        tiff_path: Path = sample_info["path"]
        
        # Read raw array directly using Path
        raw_data = tiff.imread(tiff_path).astype("float32")
        
        # Extract first N frames
        frames = raw_data[:self.n_frames]
        Nf, H, W = frames.shape
        
        # Dynamically determine image size from the image itself
        if self.crop_dim is not None and (H > self.crop_dim or W > self.crop_dim):
            start_h = (H - self.crop_dim) // 2
            start_w = (W - self.crop_dim) // 2
            frames = frames[:, start_h:start_h + self.crop_dim, start_w:start_w + self.crop_dim]
            target_dim = self.crop_dim
        else:
            # Keep native dimensions (assumes H == W for telescope images)
            target_dim = H  

        frames_tensor = torch.tensor(frames, dtype=torch.float32)

        # Build dynamic runtime config
        active_config = sample_info["config"].copy()
        active_config["target_dim"] = target_dim

        return {
            "images": frames_tensor,
            "config": active_config,
            "filename": tiff_path.name
        }

# Define an Augmentation Wrapper Class for the Training Split
class AugmentedDatasetWrapper(torch.utils.data.Dataset):
    """
    Applies consistent spatial augmentations across all frames in a stack.
    
    STRATEGY B:
    - Applies dynamic zooming to 128x128 stacks.
    - Upscales output tensors to 256x256 to prevent edge pixel loss/cropping.
    - Updates 'pixel_size' in metadata to ensure exact physical alignment with the OTF.
    """
    def __init__(self, dataset, zoom_prob: float = 0.5, zoom_range: tuple = (1.05, 1.25)):
        self.dataset = dataset
        self.zoom_prob = zoom_prob
        self.zoom_range = zoom_range

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        images = sample["images"]  # Shape: (N_frames, H, W)
        cfg = dict(sample["config"])  # Shallow copy to avoid mutating original dataset state
        
        N, H, W = images.shape
        
        # 1. Rigid Spatial Transformations (Rotations & Flips)
        angle = random.choice([0, 90, 180, 270])
        do_hflip = random.random() > 0.5
        do_vflip = random.random() > 0.5

        # 2. Strategy B Trigger: Check if this stack is 128x128 and choose dynamic zoom
        is_128 = (H == 128 and W == 128)
        zoom_factor = 1.0
        
        if is_128 and (random.random() < self.zoom_prob):
            zoom_factor = random.uniform(*self.zoom_range)  # e.g., 1.15x zoom

        augmented_frames = []
        for frame in images:
            # Reshape tensor to (1, 1, H, W) for transformation utilities
            frame_tensor = frame.unsqueeze(0).unsqueeze(0)
            
            # Apply Flips and Rotations
            if angle != 0:
                frame_tensor = TF.rotate(frame_tensor, angle)
            if do_hflip:
                frame_tensor = TF.hflip(frame_tensor)
            if do_vflip:
                frame_tensor = TF.vflip(frame_tensor)
                
            # Apply Strategy B Zoom & Rescale to 256x256
            if is_128:
                if zoom_factor > 1.0:
                    # Step A: Apply dynamic zoom by resampling to intermediate size
                    intermediate_H = int(H * zoom_factor)
                    intermediate_W = int(W * zoom_factor)
                    zoomed = F.interpolate(
                        frame_tensor, size=(intermediate_H, intermediate_W), 
                        mode='bilinear', align_corners=False
                    )
                    
                    # Step B: Rescale the zoomed FOV up to 256x256 without cropping edges
                    frame_tensor = F.interpolate(
                        zoomed, size=(256, 256), mode='bilinear', align_corners=False
                    )
                else:
                    # Standard 2x upscaling without dynamic zoom (128x128 -> 256x256)
                    frame_tensor = F.interpolate(
                        frame_tensor, size=(256, 256), mode='bilinear', align_corners=False
                    )

            augmented_frames.append(frame_tensor.squeeze())

        # Stack back to (N_frames, H_out, W_out)
        sample["images"] = torch.stack(augmented_frames, dim=0)
        
        # 3. CRITICAL PHYSICS UPDATE: Recalculate effective pixel size
        if is_128:
            total_scale_factor = 2.0 * zoom_factor  # 2.0x from grid expansion * dynamic zoom
            cfg["pixel_size"] = cfg["pixel_size"] / total_scale_factor
            cfg["target_dim"] = 256

        sample["config"] = cfg
        return sample

# if __name__ == "__main__":
#     device = 'mps'
#     n_modes = 44
#     n_frames = 5
#     pixel_size = 0.042
#     telescope_diameter = 150.0
#     central_obscuration = 0.0
#     wavelength = 8000.0
#     basis_for_wavefront = 'kl'
#     npix_image = 128

#     model = Network(device=device, n_modes=n_modes, n_frames=n_frames, pixel_size=pixel_size,
#                     telescope_diameter=telescope_diameter, central_obscuration=central_obscuration,
#                     wavelength=wavelength, basis_for_wavefront=basis_for_wavefront, npix_image=npix_image)
    
#     model = model.to(device)
    

#     images = torch.rand((2, n_frames, npix_image, npix_image), dtype=torch.float32).to(device)
#     images_ft = torch.fft.fft2(images, dim=(-2, -1))
#     variance = torch.ones((2,), dtype=torch.float32).to(device)

#     coeff, num, den, psf, otf, loss = model(images, images_ft, variance=variance)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running test pipeline on device: {device}")
    
    # -------------------------------------------------------------
    # 1. Dataset & Train / Val / Test Splitting
    # -------------------------------------------------------------
    # Update this path to point to your local testing directory
    data_path = Path("I:/Departamentos/Óptica/paulabp/master/TFM/images_networks/multiplied")
    
    # Extract only the FIRST 10 FRAMES per stack
    full_dataset = MultiTelescopeStackDataset(root_dir=data_path, n_frames=10)
    
    total_samples = len(full_dataset)
    print(f"Total dataset stacks found: {total_samples}")
    
    # Dynamic split allocation for small testing datasets (e.g. 10 total stacks)
    if total_samples < 3:
        raise ValueError("Need at least 3 dataset stacks to create Train/Val/Test splits (1 each minimum).")
        
    val_size = max(1, int(0.15 * total_samples))
    test_size = max(1, int(0.15 * total_samples))
    train_size = total_samples - val_size - test_size
    
    print(f"Split sizes -> Train: {train_size} | Val: {val_size} | Test: {test_size}")
    
    # Seed generator for reproducible splits
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Wrap ONLY the training dataset with Strategy B dynamic zooming
    train_dataset_augmented = AugmentedDatasetWrapper(
        train_dataset, 
        zoom_prob=0.5,           # 50% chance of dynamic zooming
        zoom_range=(1.05, 1.25)  # 5% - 25% zoom range
    )

    # Individual DataLoaders (batch_size=1)
    train_loader = DataLoader(train_dataset_augmented, batch_size=1, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # -------------------------------------------------------------
    # 2. Model, Optimizer & Early Stopping Setup
    # -------------------------------------------------------------
    model = Network(device=device, n_modes=44, basis_for_wavefront='kl').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_scale = 1e3
    
    # QUICK TEST CONFIGURATION: 5 Epochs & Low Patience
    num_epochs = 5            # Reduced for fast verification run
    patience = 3              # Early stopping triggers quickly if no improvement
    patience_counter = 0
    best_val_loss = float('inf')
    
    # Save directory
    save_dir = Path("./test_run_outputs")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / "best_model.pt"

    train_loss_history = []
    val_loss_history = []

    # -------------------------------------------------------------
    # 3. Training & Validation Loop
    # -------------------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        running_train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch["images"].to(device)  # Should be shape [1, 10, H, W]
            H, W = images.shape[-2], images.shape[-1]
            cfg = {k: (v[0].item() if torch.is_tensor(v[0]) else v[0]) for k, v in batch["config"].items()}

            # Updates optics dynamically based on active tensor frame shape
            model.update_telescope_basis(
                pixel_size=cfg["pixel_size"],
                telescope_diameter=cfg["telescope_diameter"],
                central_obscuration=cfg.get("central_obscuration", 0.0),
                wavelength=cfg["wavelength"],
                npix_image=H
            )
            
            # Sequence flux normalization
            seq_sum = torch.sum(images, dim=(-2, -1), keepdim=True)
            seq_mean_flux = torch.sum(seq_sum, dim=1, keepdim=True) / images.shape[1]
            images_norm = images / (seq_mean_flux + 1e-10)
            
            images_ft = torch.fft.fft2(images_norm, dim=(-2, -1), norm="ortho")
            variance = torch.tensor([1e-3], dtype=torch.float32, device=device)
            lengths = torch.tensor([images.shape[1]], dtype=torch.int64, device=device)

            optimizer.zero_grad()
            coeff, num, den, psf, otf, loss = model(images_norm, images_ft, variance, lengths=lengths)
            
            scaled_loss = loss * loss_scale
            scaled_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_loss_history.append(epoch_train_loss)

        # --- VALIDATION PHASE ---
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                H, W = images.shape[-2], images.shape[-1]
                cfg = {k: (v[0].item() if torch.is_tensor(v[0]) else v[0]) for k, v in batch["config"].items()}

                model.update_telescope_basis(
                    pixel_size=cfg["pixel_size"],
                    telescope_diameter=cfg["telescope_diameter"],
                    central_obscuration=cfg.get("central_obscuration", 0.0),
                    wavelength=cfg["wavelength"],
                    npix_image=H
                )
                
                seq_sum = torch.sum(images, dim=(-2, -1), keepdim=True)
                seq_mean_flux = torch.sum(seq_sum, dim=1, keepdim=True) / images.shape[1]
                images_norm = images / (seq_mean_flux + 1e-10)
                
                images_ft = torch.fft.fft2(images_norm, dim=(-2, -1), norm="ortho")
                variance = torch.tensor([1e-3], dtype=torch.float32, device=device)
                lengths = torch.tensor([images.shape[1]], dtype=torch.int64, device=device)

                _, _, _, _, _, loss = model(images_norm, images_ft, variance, lengths=lengths)
                running_val_loss += loss.item()

        epoch_val_loss = running_val_loss / len(val_loader)
        val_loss_history.append(epoch_val_loss)

        print(f"Test Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {epoch_train_loss:.6e} | Val Loss: {epoch_val_loss:.6e}")

        # --- CHECKPOINTING ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, best_model_path)
            print(f"  --> Checkpoint saved at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered early during testing at epoch {epoch}.")
                break

    # -------------------------------------------------------------
    # 4. Save Test Outputs
    # -------------------------------------------------------------
    history = {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "best_val_loss": best_val_loss
    }
    with open(save_dir / "test_loss_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    plt.figure(figsize=(8, 4))
    plt.plot(train_loss_history, label="Training Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Sanity Check Test Run")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "test_loss_plot.png", dpi=150)
    plt.close()
    
    print(f"✅ Sanity check complete. Results saved to {save_dir.resolve()}")