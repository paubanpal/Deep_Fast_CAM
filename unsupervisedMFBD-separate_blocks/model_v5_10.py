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
import tifffile as tiff
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')  # Headless backend for HPC clusters
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
        super().__init__()

        self.n_lstm = n_lstm

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

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.C41 = nn.Conv2d(n, self.n_lstm, kernel_size=1, stride=1)

    def weights_init(self):
        for module in self.modules():
            kaiming_init(module)

    def forward(self, images):
        if images.dim() == 4:
            images = images.unsqueeze(2)

        B, Nf, C, H, W = images.shape
        tmp = images.view(B * Nf, C, H, W)

        A01 = self.A01(tmp)

        C01 = self.C01(A01)
        C02 = self.C02(C01)
        C03 = self.C03(C02)
        C04 = C01 + self.C04(C03)

        C11 = self.C11(C04)
        C12 = self.C12(C11)
        C13 = self.C13(C12)
        C14 = C11 + self.C14(C13)

        C21 = self.C21(C14)
        C22 = self.C22(C21)
        C23 = self.C23(C22)
        C24 = C21 + self.C24(C23)

        out = self.global_pool(C24)
        out = self.C41(out)

        out = torch.flatten(out, start_dim=1)
        out = out.view(B, Nf, self.n_lstm)

        return out
    
class LSTM(nn.Module):
    def __init__(self, n_modes, n_lstm):
        super().__init__()

        self.n_modes = n_modes
        self.n_lstm = n_lstm
        
        self.C42 = nn.Linear(2*self.n_lstm, self.n_lstm)
        self.C43 = nn.Linear(self.n_lstm, n_modes)
        
        self.elu = nn.ELU()

        self.lstm = nn.LSTM(self.n_lstm, self.n_lstm, batch_first=True, bidirectional=True, dropout=0.0)

    def weights_init(self):
        for module in self.modules():
            kaiming_init(module)

        nn.init.normal_(self.C43.weight, std=1e-3)
        if self.C43.bias is not None:
            nn.init.zeros_(self.C43.bias)

    def forward(self, latent_features, lengths=None):
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                latent_features, lengths.to(torch.int64).cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, _ = self.lstm(latent_features)

        out = out.reshape(-1, 2 * self.n_lstm)
        out = self.elu(self.C42(out))
        out = self.C43(out)
        return out

class Network(nn.Module):
    def __init__(self, device='cpu', n_modes=44, n_frames=50, pixel_size=0.042, 
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
        self.current_config = None

        print(f"Wavelength : {self.wavelength} A")
        print(f"Diameter : {self.telescope_diameter} cm")
        print(f"Central obscuration : {self.central_obscuration} cm")
        print(f"Pixel size : {self.pixel_size} arcsec")

        self.overfill = util.psf_scale(self.wavelength, self.telescope_diameter, self.pixel_size)                
        if (self.overfill < 1.0):
            raise Exception(f"The pixel size is not small enough to model a telescope with D={self.telescope_diameter} cm")
            
        pupil = util.aperture(npix=self.npix_image, cent_obs = self.central_obscuration / self.telescope_diameter, spider=0, overfill=self.overfill)
        pupil = torch.tensor(pupil.astype('float32'))
            
        # if (self.basis_for_wavefront == 'zernike'):
        #     print("Computing Zernike modes...")
        #     Z_machine = zern.ZernikeNaive(mask=[])
        #     x = np.linspace(-1, 1, self.npix_image)
        #     xx, yy = np.meshgrid(x, x)
        #     rho = self.overfill * np.sqrt(xx ** 2 + yy ** 2)
        #     theta = np.arctan2(yy, xx)
        #     aperture_mask = rho <= 1.0

        #     basis = np.zeros((self.n_modes, self.npix_image, self.npix_image))
            
        #     for j in range(self.n_modes):
        #         n, m = zern.zernIndex(j+2)
        #         Z = Z_machine.Z_nm(n, m, rho, theta, True, 'Jacobi')
        #         basis[j,:,:] = Z * aperture_mask

        # if (self.basis_for_wavefront == 'kl'):
        #     print("Computing KL modes...")
        #     kl = kl_modes.KL()
        #     basis = kl.precalculate_covariance(npix_image = self.npix_image, n_modes_max = self.n_modes, first_noll = 1, overfill=self.overfill)

        # zeros = torch.zeros((self.npix_image, self.npix_image, 1), dtype=torch.float32)

        # self.register_buffer('zeros', zeros)
        # self.register_buffer('pupil', pupil)
        # self.register_buffer('basis', torch.tensor(basis.astype('float32')))

        # Leave buffer initialization deferred to update_telescope_basis()
        self.basis_cache = {}

        self.cnn = CNN(n=16, n_lstm=256)
        self.cnn.weights_init()

        self.lstm = LSTM(n_modes=self.n_modes, n_lstm=256)
        self.lstm.weights_init()

    def update_telescope_basis(self, pixel_size, telescope_diameter, central_obscuration, wavelength, npix_image):
        # Initialize cache on the module if it doesn't exist yet
        if not hasattr(self, 'basis_cache'):
            self.basis_cache = {}

        # Insert at the beginning of update_telescope_basis:
        if len(self.basis_cache) > 20:
            self.basis_cache.clear()
            torch.cuda.empty_cache()

        # Round floating point values slightly to prevent tiny float rounding errors from missing the cache
        pixel_size_key = round(float(pixel_size), 6)
    
        config_key = (pixel_size_key, telescope_diameter, central_obscuration, wavelength, int(npix_image))
    
        if getattr(self, 'current_config', None) == config_key:
            return  # Already configured for this exact state on the current step

        self.pixel_size = pixel_size
        self.telescope_diameter = telescope_diameter
        self.central_obscuration = central_obscuration
        self.wavelength = wavelength
        self.npix_image = int(npix_image)
        self.current_config = config_key

        # Check if we already computed and cached this exact configuration earlier
        if config_key in self.basis_cache:
            cached_pupil, cached_basis, cached_zeros = self.basis_cache[config_key]
            self.register_buffer('pupil', cached_pupil.to(self.device), persistent=False)
            self.register_buffer('basis', cached_basis.to(self.device), persistent=False)
            self.register_buffer('zeros', cached_zeros.to(self.device), persistent=False)
            return

        # --- Compute from scratch only if NOT in cache ---
        self.overfill = util.psf_scale(self.wavelength, self.telescope_diameter, self.pixel_size)                 
        if self.overfill < 1.0:
            raise Exception(f"Pixel size {self.pixel_size} arcsec is not small enough to model D={self.telescope_diameter} cm")
        
        pupil = util.aperture(npix=self.npix_image, cent_obs=self.central_obscuration / self.telescope_diameter, spider=0, overfill=self.overfill)
        pupil = torch.tensor(pupil.astype('float32'), device=self.device)
        
        if self.basis_for_wavefront == 'zernike':
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

        elif self.basis_for_wavefront == 'kl':
            kl = kl_modes.KL()
            basis = kl.precalculate_covariance(npix_image=self.npix_image, n_modes_max=self.n_modes, first_noll=1, overfill=self.overfill)

        zeros = torch.zeros((self.npix_image, self.npix_image, 1), dtype=torch.float32, device=self.device)
        basis_tensor = torch.tensor(basis.astype('float32'), device=self.device)

        # Store tensors in dictionary cache
        self.basis_cache[config_key] = (pupil, basis_tensor, zeros)

        # Register as active PyTorch buffers
        self.register_buffer('zeros', zeros, persistent=False)
        self.register_buffer('pupil', pupil, persistent=False)
        self.register_buffer('basis', basis_tensor, persistent=False)

    def compute_psfs(self, coeff):
        wavefront = torch.einsum('ij,jkl->ikl', coeff, self.basis)
        phase = self.pupil[None, :, :] * torch.exp(1j * wavefront)

        ft = torch.fft.fft2(phase, norm="ortho")
        psf = (torch.conj(ft) * ft).real
        # psf_sum = torch.clamp(torch.sum(psf, [-1, -2], keepdim=True), min=1e-8)
        psf_sum = torch.sum(psf, [-1, -2], keepdim=True) + 1e-8
        psf_norm = psf / psf_sum
        otf = torch.fft.fft2(psf_norm, norm="ortho")

        return psf, otf, wavefront

    def loss_and_wiener_filter(self, im_ft, psf_ft, variance, lengths=None):
        if lengths is not None:
            Nf = psf_ft.shape[1]
            lengths_dev = lengths.to(psf_ft.device)
            mask = (torch.arange(Nf, device=psf_ft.device)[None, :] < lengths_dev[:, None])[:, :, None, None]
            psf_ft = psf_ft * mask
            im_ft = im_ft * mask

        S_star_D = torch.conj(psf_ft) * im_ft
        D_star_S = torch.conj(im_ft) * psf_ft
        modulus_S = torch.conj(psf_ft) * psf_ft
        modulus_D = torch.conj(im_ft) * im_ft
        
        sum_D_star_S = torch.sum(D_star_S, dim=1)
        modulus_D_star_S = torch.conj(sum_D_star_S) * sum_D_star_S

        denominator = torch.sum(modulus_S, dim=1)
        numerator = torch.sum(S_star_D, dim=1)

        tmp = torch.sum(modulus_D, dim=1)
        #loss = tmp - modulus_D_star_S / torch.clamp(variance[:, None, None] + denominator, min=1e-8)
        # Extract real components and safely clamp the real denominator
        
        # denom_real = (variance[:, None, None] + denominator).real
        # denom_safe = torch.clamp(denom_real, min=1e-8)
        # loss = tmp.real - (modulus_D_star_S.real / denom_safe)

        # Use explicit epsilon addition instead of hard clamping
        eps = 1e-6
        denom_safe = denominator.real + variance[:, None, None] + eps
        loss = tmp.real - (modulus_D_star_S.real / denom_safe)

        if lengths is not None:
            total_valid_frames = torch.sum(lengths_dev)
            loss_mn = (torch.sum(loss.real) / total_valid_frames)
        else:
            loss_mn = torch.mean(loss.real)

        return numerator, denominator, loss_mn

    def forward(self, images, images_ft, variance, lengths=None):
        B, Nf = images.shape[0], images.shape[1]

        latent_features = self.cnn(images)
        coeff = self.lstm(latent_features, lengths=lengths)

        tmp = rearrange(coeff, '(b f) m -> b f m', f=Nf, m=self.n_modes)

        if lengths is not None:
            lengths_dev = lengths.to(images.device)
            mask = torch.arange(Nf, device=images.device)[None, :] < lengths_dev[:, None]
            mask_expanded = mask.unsqueeze(-1)

            sum_coeff = torch.sum(tmp * mask_expanded, dim=1)
            avg = sum_coeff / lengths_dev[:, None]
        else:
            avg = torch.mean(tmp, dim=1)

        mask_tt = torch.zeros_like(avg)
        mask_tt[:, :2] = 1.0
        avg = avg * mask_tt

        avg = repeat(avg, 'b m -> b f m', f=Nf)
        avg = rearrange(avg, 'b f m -> (b f) m')

        coeff_corrected = coeff - avg
        psf, psf_ft, wavefront = self.compute_psfs(coeff_corrected)
        psf_ft = rearrange(psf_ft, '(b f) x y -> b f x y', f=Nf)

        numerator, denominator, loss = self.loss_and_wiener_filter(images_ft, psf_ft, variance, lengths=lengths)
        
        return coeff_corrected, numerator, denominator, psf, psf_ft, loss

class DynamicStackDataset(Dataset):
    """
    Manages loading and dynamic slicing of random stacks.
    Loads a stack and extracts a contiguous block of N consecutive frames.
    Selects validation/test samples based on shortest frame length to preserve longer stacks for training.
    """
    def __init__(self, root_dir: str | Path, crop_dim: int | None = None, seed: int = 42):
        self.root_path = Path(root_dir)
        self.crop_dim = crop_dim
        
        telescope_samples = {}
        for tel_dir in sorted(self.root_path.iterdir()):
            if not tel_dir.is_dir():
                continue
                
            config_path = tel_dir / "config.json"
            if not config_path.exists():
                print(f"Warning: Skipping {tel_dir.name} because config.json was not found.")
                continue
                
            with open(config_path, "r", encoding="utf-8") as f:
                tel_config = json.load(f)
                
            tiff_files = sorted(list(tel_dir.glob("*.tiff")) + list(tel_dir.glob("*.tif")))
            telescope_samples[tel_dir.name] = []
            for tiff_path in tiff_files:
                # Read tiff header to check total frame count
                with tiff.TiffFile(tiff_path) as tf:
                    total_frames = len(tf.pages)

                telescope_samples[tel_dir.name].append({
                    "path": tiff_path,
                    "config": tel_config,
                    "n_frames": total_frames
                })

        self.train_samples = []
        self.val_samples = []
        self.test_samples = []

        # Split 1 val and 1 test stack from EACH telescope, prioritizing the shortest stacks
        for tel_name, samples in telescope_samples.items():
            if len(samples) < 3:
                raise ValueError(f"Telescope directory {tel_name} has less than 3 stacks!")
                
            # Sort ascending by frame count: shortest stacks selected for Val/Test
            sorted_samples = sorted(samples, key=lambda s: s["n_frames"])
            
            self.val_samples.append(sorted_samples[0])
            self.test_samples.append(sorted_samples[1])
            self.train_samples.extend(sorted_samples[2:])

        print(f"Discovered {sum(len(s) for s in telescope_samples.values())} stack files across telescope subdirectories.")
        print(f"Split -> Train: {len(self.train_samples)} | Val: {len(self.val_samples)} | Test: {len(self.test_samples)}")
        for vs in self.val_samples:
            print(f"  [Val Target] {vs['path'].name} | Total Frames: {vs['n_frames']}")

    def sample_slice(self, sample_info: dict, n_frames: int = 50, start_idx: int | None = None):
        """
        Extracts a N-frame contiguous slice from a given sample dictionary.
        If start_idx is None, samples randomly (training).
        Pass start_idx=0 for deterministic validation from frame 0.
        """
        tiff_path: Path = sample_info["path"]
        
        # Read complete stack array
        raw_data = tiff.imread(tiff_path).astype("float32")
        total_frames, H, W = raw_data.shape

        if total_frames < n_frames:
            raise ValueError(f"Stack {tiff_path.name} has only {total_frames} frames, but {n_frames} were requested.")

        max_start = total_frames - n_frames
        
        if start_idx is None:
            start_idx = random.randint(0, max_start)
        else:
            start_idx = min(start_idx, max_start)
        
        frames = raw_data[start_idx : start_idx + n_frames]
        
        # Apply crop if necessary
        if self.crop_dim is not None and (H > self.crop_dim or W > self.crop_dim):
            start_h = (H - self.crop_dim) // 2
            start_w = (W - self.crop_dim) // 2
            frames = frames[:, start_h:start_h + self.crop_dim, start_w:start_w + self.crop_dim]
            target_dim = self.crop_dim
        else:
            target_dim = H

        frames_tensor = torch.tensor(frames, dtype=torch.float32)

        active_config = sample_info["config"].copy()
        active_config["target_dim"] = target_dim

        return {
            "images": frames_tensor,
            "config": active_config,
            "filename": tiff_path.name,
            "start_frame": start_idx
        }

class AugmentedDatasetWrapper:
    """
    Applies consistent spatial augmentations across all frames in a sampled sequence.
    """
    def __init__(self, zoom_prob: float = 0.5, zoom_range: tuple = (1.05, 1.25)):
        self.zoom_prob = zoom_prob
        self.zoom_range = zoom_range

    def augment(self, sample: dict) -> dict:
        images = sample["images"]  # Shape: (N_frames, H, W)
        cfg = dict(sample["config"])
        
        N, H, W = images.shape
        
        angle = random.choice([0, 90, 180, 270])
        do_hflip = random.random() > 0.5
        do_vflip = random.random() > 0.5

        is_128 = (H == 128 and W == 128)
        zoom_factor = 1.0
        
        # Decide if we apply zoom (only eligible for 128x128 images)
        if is_128 and (random.random() < self.zoom_prob):
            zoom_factor = random.uniform(*self.zoom_range)

        augmented_frames = []
        for frame in images:
            frame_tensor = frame.unsqueeze(0).unsqueeze(0)
            
            # 1. Rotations & Flips
            if angle != 0:
                frame_tensor = TF.rotate(frame_tensor, angle)
            if do_hflip:
                frame_tensor = TF.hflip(frame_tensor)
            if do_vflip:
                frame_tensor = TF.vflip(frame_tensor)
                
            # 2. Conditional Resizing: ONLY if zoomed in
            if is_128 and zoom_factor > 1.0:
                intermediate_H = int(H * zoom_factor)
                intermediate_W = int(W * zoom_factor)
                zoomed = F.interpolate(
                    frame_tensor, size=(intermediate_H, intermediate_W), 
                    mode='bilinear', align_corners=False
                )
                # Scale up to 256x256 to accommodate the zoomed-in ROI
                frame_tensor = F.interpolate(
                    zoomed, size=(256, 256), mode='bilinear', align_corners=False
                )

            augmented_frames.append(frame_tensor.squeeze())

        sample["images"] = torch.stack(augmented_frames, dim=0)
        
        # 3. Update configuration metadata ONLY when zoomed in
        if is_128 and zoom_factor > 1.0:
            total_scale_factor = 2.0 * zoom_factor
            cfg["pixel_size"] = cfg["pixel_size"] / total_scale_factor
            cfg["target_dim"] = 256

        sample["config"] = cfg
        return sample

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data_path = Path("/scratch/paulabp/TFM/images/images_for_network/originals_cropped")
    dataset = DynamicStackDataset(root_dir=data_path, seed=42)
    augmentor = AugmentedDatasetWrapper(zoom_prob=0.5, zoom_range=(1.05, 1.25))

    # -------------------------------------------------------------
    # Model, Optimizer & Early Stopping Setup
    # -------------------------------------------------------------
    n_frames_per_epoch = 10
    model = Network(device=device, n_modes=119, n_frames=n_frames_per_epoch, basis_for_wavefront='kl').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_scale = 1.0
    
    patience = 50
    patience_counter = 0
    best_val_loss = float('inf')
    num_epochs = 100
    
    save_dir = Path("/scratch/paulabp/TFM/run_outputs_90")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / "best_model.pt"

    train_loss_history = []
    val_loss_history = []

    # -------------------------------------------------------------
    # Epoch Loop: Process All Training Stacks & Val Stacks per Epoch
    # -------------------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        train_epoch_losses = []
        
        train_samples_shuffled = list(dataset.train_samples)
        random.shuffle(train_samples_shuffled)
        
        for sample_info in train_samples_shuffled:
            # Random contiguous 50-frame slice for training
            sample = dataset.sample_slice(sample_info, n_frames=n_frames_per_epoch, start_idx=None)
            augmented_sample = augmentor.augment(sample)
            
            images = augmented_sample["images"].unsqueeze(0).to(device)  # Add Batch Dim -> [1, 50, H, W]
            cfg = augmented_sample["config"]
            H, W = images.shape[-2], images.shape[-1]

            model.update_telescope_basis(
                pixel_size=cfg["pixel_size"],
                telescope_diameter=cfg["telescope_diameter"],
                central_obscuration=cfg.get("central_obscuration", 0.0),
                wavelength=cfg["wavelength"],
                npix_image=H
            )
            
            seq_sum = torch.sum(images, dim=(-2, -1), keepdim=True)
            seq_mean_flux = torch.sum(seq_sum, dim=1, keepdim=True) / images.shape[1]
            # images_norm = images / torch.clamp(seq_mean_flux, min=1e-8)
            images_norm = images / (seq_mean_flux + 1e-8)
            
            images_ft = torch.fft.fft2(images_norm, dim=(-2, -1), norm="ortho")
            variance = torch.tensor([1e-3], dtype=torch.float32, device=device)
            lengths = torch.tensor([images.shape[1]], dtype=torch.int64, device=device)

            optimizer.zero_grad()
            coeff, num, den, psf, otf, train_loss = model(images_norm, images_ft, variance, lengths=lengths)
            
            scaled_loss = train_loss * loss_scale
            scaled_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_epoch_losses.append(train_loss.item())

        train_loss_val = float(np.mean(train_epoch_losses))
        train_loss_history.append(train_loss_val)

        # --- VALIDATION PHASE (1 shortest stack per telescope, deterministic frames 0..49, unaugmented) ---
        model.eval()
        val_epoch_losses = []
        with torch.no_grad():
            for sample_info in dataset.val_samples:
                # Deterministic evaluation using first 50 frames (start_idx=0)
                val_sample = dataset.sample_slice(sample_info, n_frames=n_frames_per_epoch, start_idx=0)
                val_images = val_sample["images"].unsqueeze(0).to(device)
                val_cfg = val_sample["config"]
                val_H = val_images.shape[-2]

                model.update_telescope_basis(
                    pixel_size=val_cfg["pixel_size"],
                    telescope_diameter=val_cfg["telescope_diameter"],
                    central_obscuration=val_cfg.get("central_obscuration", 0.0),
                    wavelength=val_cfg["wavelength"],
                    npix_image=val_H
                )
                
                val_seq_sum = torch.sum(val_images, dim=(-2, -1), keepdim=True)
                val_seq_mean_flux = torch.sum(val_seq_sum, dim=1, keepdim=True) / val_images.shape[1]
                val_images_norm = val_images / (val_seq_mean_flux + 1e-8)
                
                val_images_ft = torch.fft.fft2(val_images_norm, dim=(-2, -1), norm="ortho")

                _, _, _, _, _, val_loss = model(val_images_norm, val_images_ft, variance, lengths=lengths)
                val_epoch_losses.append(val_loss.item())

        val_loss_val = float(np.mean(val_epoch_losses))
        val_loss_history.append(val_loss_val)

        print(f"Epoch {epoch:03d}/{num_epochs} | Train Loss: {train_loss_val:.6e} | Val Loss: {val_loss_val:.6e}")

        # --- CHECKPOINTING ---
        if val_loss_val < best_val_loss:
            best_val_loss = val_loss_val
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, best_model_path)
            print(f"--> Saved best model checkpoint (Val Loss: {best_val_loss:.6e})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    # Save training logs and plots
    history = {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "best_val_loss": best_val_loss
    }
    with open(save_dir / "loss_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    plt.figure(figsize=(9, 5))
    plt.plot(train_loss_history, label="Training Loss", color="#1f77b4", linewidth=2)
    plt.plot(val_loss_history, label="Validation Loss", color="#ff7f0e", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MOMFBD Loss")
    plt.title("Dynamic 50-Frame Stack Training")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "loss_plot.png", dpi=300)
    plt.close()

    print(f"Finished! Run outputs written to {save_dir.resolve()}")