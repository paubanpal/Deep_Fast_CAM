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
import traceback
import sys

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

        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, bias=not bn)
        self.reflection = nn.ReflectionPad2d(kernel_size // 2)

        if self.use_bn:
            # In pre-activation, normalization operates on the input channels ('inplanes')
            self.bn = nn.InstanceNorm2d(inplanes, affine=True)

        if self.use_activation:
            self.elu = nn.ELU(inplace=False)

    def forward(self, x):
        out = x
        if self.use_bn:
            out = self.bn(out)
            if self.use_activation:
                out = self.elu(out)
        elif self.use_activation:
            out = self.elu(out)

        out = self.reflection(out)
        out = self.conv(out)

        return out
    
class CNN(nn.Module):
    def __init__(self, n, n_lstm):
        super().__init__()

        self.n_lstm = n_lstm

        # 2-channel input: Magnitude + Phase
        self.A01 = ConvBlock(2, n, kernel_size=9, bn=False, activation=False)

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
        if images.dim() == 5:
            B, Nf, C, H, W = images.shape
            tmp = images.view(B * Nf, C, H, W)
        elif images.dim() == 4:
            tmp = images
            B, Nf = images.shape[0], 1
        else:
            raise ValueError(f"Unsupported input dimension: {images.dim()}")

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

        #nn.init.normal_(self.C43.weight, std=1e-3)
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
            
        # Leave buffer initialization deferred to update_telescope_basis()
        self.pupil = None
        self.basis = None
        self.zeros = None
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
            self.pupil = cached_pupil.to(self.device)
            self.basis = cached_basis.to(self.device)
            self.zeros = cached_zeros.to(self.device)
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

        # Assign active PyTorch buffers
        self.pupil = pupil
        self.basis = basis_tensor
        self.zeros = zeros

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

        # Use explicit epsilon addition instead of hard clamping
        eps = 1e-4
        denom_safe = torch.clamp(denominator.real, min=0.0) + variance[:, None, None] + eps
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
    Loads Mapped Magnitude, Phase, and Original Real image paths specified in mapping_fixed.json within root_dir.
    Supported JSON structure: 
    "magnitude.tif": {"phase": "phase.tif", "original": "original.tif"}
    """
    def __init__(self, root_dir: str | Path, originals_dir: str | Path, crop_dim: int | None = None, seed: int = 42):
        self.root_path = Path(root_dir)
        self.originals_path = Path(originals_dir)
        self.crop_dim = crop_dim
        
        fft_map = {}
        mapping_path = self.root_path / "mapping_fixed.json"
        if mapping_path.exists():
            with open(mapping_path, "r", encoding="utf-8") as f:
                fft_map = json.load(f)
            print(f"[INFO] Successfully loaded mapping_fixed.json with {len(fft_map)} mapped telescopes.")
        else:
            print(f"Warning: mapping_fixed.json not found in {self.root_path}.")

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
                
            tel_fft_map = fft_map.get(tel_dir.name, {})
            telescope_samples[tel_dir.name] = []

            for module_file, target_info in tel_fft_map.items():
                module_path = tel_dir / module_file

                # Obtención de nombres de archivo desde mapping_fixed.json
                if isinstance(target_info, dict):
                    phase_file = target_info.get("phase", module_file)
                    orig_file = target_info.get("original", module_file)
                elif isinstance(target_info, list):
                    phase_file = target_info[0]
                    orig_file = target_info[1]
                else:
                    phase_file = target_info
                    orig_file = module_file

                phase_path = tel_dir / phase_file
                
                # Búsqueda del archivo de la imagen original en el directorio de originales asignado por código
                orig_file_path = self.originals_path / tel_dir.name / orig_file
                if not orig_file_path.exists():
                    orig_file_path = self.originals_path / orig_file

                if not module_path.exists():
                    print(f"Warning: Module file does not exist: {module_path}. Skipping.")
                    continue
                if not phase_path.exists():
                    print(f"Warning: Phase file does not exist: {phase_path}. Skipping.")
                    continue

                try:
                    with tiff.TiffFile(module_path) as tf:
                        total_frames = len(tf.pages)
                except Exception as e:
                    print(f"[ERROR] Failed to read {module_path.name}: {e}")
                    continue

                telescope_samples[tel_dir.name].append({
                    "module_path": module_path,
                    "phase_path": phase_path,
                    "orig_path": orig_file_path,
                    "config": tel_config,
                    "n_frames": total_frames,
                    "filename": module_file
                })

        self.train_samples = []
        self.val_samples = []
        self.test_samples = []

        # Split 1 val and 1 test stack from EACH telescope, prioritizing the shortest stacks
        for tel_name, samples in telescope_samples.items():
            if len(samples) < 3:
                print(f"Warning: Telescope {tel_name} has fewer than 3 valid stacks.")
                continue
                
            sorted_samples = sorted(samples, key=lambda s: s["n_frames"])
            
            self.val_samples.append(sorted_samples[0])
            self.test_samples.append(sorted_samples[1])
            self.train_samples.extend(sorted_samples[2:])

        print(f"Discovered {sum(len(s) for s in telescope_samples.values())} stack files across telescope subdirectories.")
        print(f"Split -> Train: {len(self.train_samples)} | Val: {len(self.val_samples)} | Test: {len(self.test_samples)}")
        for vs in self.val_samples:
            print(f"  [Val Target] {vs['module_path'].name} | Total Frames: {vs['n_frames']}")

        if len(self.train_samples) == 0:
            raise RuntimeError("[CRITICAL ERROR] Training sample list is empty. Check mapping_fixed.json.")

    def sample_slice(self, sample_info: dict, n_frames: int = 50, start_idx: int | None = None):
        """
        Extracts a N-frame contiguous slice from a given sample dictionary.
        Loads Magnitude and Phase for CNN input, and Original Spatial Image to compute FFT2 for Wiener Loss.
        """
        module_path: Path = sample_info["module_path"]
        phase_path: Path = sample_info["phase_path"]
        orig_path: Path = sample_info["orig_path"]
        
        # Read complete stack array for module and phase
        raw_module = tiff.imread(module_path).astype("float32")
        total_frames, H, W = raw_module.shape

        if total_frames < n_frames:
            raise ValueError(f"Stack {module_path.name} has only {total_frames} frames, but {n_frames} were requested.")

        max_start = total_frames - n_frames
        
        if start_idx is None:
            start_idx = random.randint(0, max_start)
        else:
            start_idx = min(start_idx, max_start)
        
        mod_slice = raw_module[start_idx : start_idx + n_frames]
        raw_phase = tiff.imread(phase_path)[start_idx : start_idx + n_frames].astype("float32")
        
        mod_tensor = torch.tensor(mod_slice, dtype=torch.float32)
        phase_tensor = torch.tensor(raw_phase, dtype=torch.float32)

        # Carga la imagen real original desde la ruta asignada
        if orig_path.exists():
            raw_orig = tiff.imread(orig_path)[start_idx : start_idx + n_frames].astype("float32")
            orig_tensor = torch.tensor(raw_orig, dtype=torch.float32)
        else:
            print(f"Warning: Original file not found at {orig_path}. Falling back to magnitude tensor.")
            orig_tensor = mod_tensor

        # Normalización de la imagen original por su valor píxel máximo del stack
        orig_max = orig_tensor.amax(dim=(-2, -1), keepdim=True)
        orig_tensor = orig_tensor / (orig_max + 1e-8)

        # Apply crop if necessary
        if self.crop_dim is not None and (H > self.crop_dim or W > self.crop_dim):
            start_h = (H - self.crop_dim) // 2
            start_w = (W - self.crop_dim) // 2
            mod_tensor = mod_tensor[:, start_h:start_h + self.crop_dim, start_w:start_w + self.crop_dim]
            phase_tensor = phase_tensor[:, start_h:start_h + self.crop_dim, start_w:start_w + self.crop_dim]
            orig_tensor = orig_tensor[:, start_h:start_h + self.crop_dim, start_w:start_w + self.crop_dim]
            target_dim = self.crop_dim
        else:
            target_dim = H

        # Preprocesamiento de entrada de 2 canales para la red (Módulo y Fase de la imagen original)
        dc_val = orig_tensor[:, 0:1, 0:1]
        orig_tensor_norm = orig_tensor / (dc_val + 1e-8)
        orig_tensor_cnn = torch.log1p(torch.clamp(orig_tensor_norm, min=0.0))
        
        # Calcular fase de la imagen original mediante la FFT de la imagen original
        fft_complex = torch.fft.fft2(orig_tensor, norm="ortho")
        orig_phase = torch.angle(fft_complex)

        input_2ch = torch.stack([orig_tensor_cnn, orig_phase], dim=1)

        active_config = sample_info["config"].copy()
        active_config["target_dim"] = target_dim

        return {
            "images": input_2ch,            # [N_frames, 2, H, W]
            "images_ft": fft_complex,       # [N_frames, H, W] (complex)
            "config": active_config,
            "filename": sample_info["filename"],
            "start_frame": start_idx
        }

class AugmentedDatasetWrapper:
    """
    Applies consistent spatial augmentations across all frames in a sampled sequence.
    """
    def __init__(self, zoom_prob: float = 0.0, zoom_range: tuple = (1.05, 1.25)):
        self.zoom_prob = zoom_prob
        self.zoom_range = zoom_range

    def augment(self, sample: dict) -> dict:
        images = sample["images"]        # Shape: (N_frames, 2, H, W)
        images_ft = sample["images_ft"]  # Shape: (N_frames, H, W) (complex)
        cfg = dict(sample["config"])
        
        N, C, H, W = images.shape
        
        angle = random.choice([0, 90, 180, 270])
        do_hflip = random.random() > 0.5
        do_vflip = random.random() > 0.5

        is_128 = (H == 128 and W == 128)
        zoom_factor = 1.0
        
        # Decide if we apply zoom (only eligible for 128x128 images)
        if is_128 and (random.random() < self.zoom_prob):
            zoom_factor = random.uniform(*self.zoom_range)

        augmented_frames = []
        for frame_2ch in images:
            # 1. Rotations & Flips
            if angle != 0:
                frame_2ch = TF.rotate(frame_2ch, angle)
            if do_hflip:
                frame_2ch = TF.hflip(frame_2ch)
            if do_vflip:
                frame_2ch = TF.vflip(frame_2ch)
                
            # 2. Conditional Resizing: ONLY if zoomed in
            if is_128 and zoom_factor > 1.0:
                intermediate_H = int(H * zoom_factor)
                intermediate_W = int(W * zoom_factor)
                zoomed = F.interpolate(
                    frame_2ch.unsqueeze(0), size=(intermediate_H, intermediate_W), 
                    mode='bilinear', align_corners=False
                )
                # Scale up to 256x256 to accommodate the zoomed-in ROI
                frame_2ch = F.interpolate(
                    zoomed, size=(256, 256), mode='bilinear', align_corners=False
                ).squeeze(0)

            augmented_frames.append(frame_2ch)

        sample["images"] = torch.stack(augmented_frames, dim=0)
        
        # Direct spatial transformations on the complex FFT tensor (images_ft) via Real and Imaginary components
        real_ft = images_ft.real
        imag_ft = images_ft.imag

        if angle != 0:
            real_ft = TF.rotate(real_ft, angle)
            imag_ft = TF.rotate(imag_ft, angle)
        if do_hflip:
            real_ft = TF.hflip(real_ft)
            imag_ft = TF.hflip(imag_ft)
        if do_vflip:
            real_ft = TF.vflip(real_ft)
            imag_ft = TF.vflip(imag_ft)

        if is_128 and zoom_factor > 1.0:
            intermediate_H = int(H * zoom_factor)
            intermediate_W = int(W * zoom_factor)
            
            real_zoomed = F.interpolate(
                real_ft.unsqueeze(1), size=(intermediate_H, intermediate_W), 
                mode='bilinear', align_corners=False
            )
            imag_zoomed = F.interpolate(
                imag_ft.unsqueeze(1), size=(intermediate_H, intermediate_W), 
                mode='bilinear', align_corners=False
            )
            
            real_ft = F.interpolate(
                real_zoomed, size=(256, 256), mode='bilinear', align_corners=False
            ).squeeze(1)
            imag_ft = F.interpolate(
                imag_zoomed, size=(256, 256), mode='bilinear', align_corners=False
            ).squeeze(1)

        sample["images_ft"] = torch.complex(real_ft, imag_ft)

        # 3. Update configuration metadata ONLY when zoomed in
        if is_128 and zoom_factor > 1.0:
            total_scale_factor = 2.0 * zoom_factor
            cfg["pixel_size"] = cfg["pixel_size"] / total_scale_factor
            cfg["target_dim"] = 256

        sample["config"] = cfg
        return sample

if __name__ == "__main__":
    # Disable output buffering to ensure print statements write immediately to Slurm/HPC log files
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Active device: {device}")
        
        # Ruta del directorio de Módulo y Fase
        data_path = Path("/scratch/paulabp/TFM/images/images_for_network/FFT/originals/FFTs")
        
        # Directorio de las imágenes reales originales especificadas por código
        originals_dir = Path("/scratch/paulabp/TFM/images/images_for_network/originals_cropped")
        
        print(f"[INFO] Loading dataset from: {data_path.resolve()}")
        print(f"[INFO] Loading real original images from: {originals_dir.resolve()}")
        
        dataset = DynamicStackDataset(root_dir=data_path, originals_dir=originals_dir, seed=42)
        augmentor = AugmentedDatasetWrapper(zoom_prob=0.0, zoom_range=(1.05, 1.25))

        # -------------------------------------------------------------
        # Gradient Accumulation Configuration
        # -------------------------------------------------------------
        num_train_samples = len(dataset.train_samples) # 10
        accumulation_steps = max(1, num_train_samples)

        # -------------------------------------------------------------
        # Model, Optimizer & Early Stopping Setup
        # -------------------------------------------------------------
        n_frames_per_epoch = 50
        model = Network(device=device, n_modes=119, n_frames=n_frames_per_epoch, basis_for_wavefront='kl').to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        num_epochs = 100
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        loss_scale = 1.0
        
        patience = 50
        patience_counter = 0
        best_val_loss = float('inf')
        
        save_dir = Path("/scratch/paulabp/TFM/run_outputs_v6_50_acc_sched_instance_norm_2channels_1dir_fixed_v8")
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
            
            optimizer.zero_grad()

            for step, sample_info in enumerate(train_samples_shuffled, start=1):
                # Random contiguous 50-frame slice for training
                sample = dataset.sample_slice(sample_info, n_frames=n_frames_per_epoch, start_idx=None)
                augmented_sample = augmentor.augment(sample)
                
                images_2ch = augmented_sample["images"].unsqueeze(0).to(device)  # Shape: [1, N_frames, 2, H, W]
                images_ft = augmented_sample["images_ft"].unsqueeze(0).to(device) # Shape: [1, N_frames, H, W]
                cfg = augmented_sample["config"]
                H, W = images_2ch.shape[-2], images_2ch.shape[-1]

                model.update_telescope_basis(
                    pixel_size=cfg["pixel_size"],
                    telescope_diameter=cfg["telescope_diameter"],
                    central_obscuration=cfg.get("central_obscuration", 0.0),
                    wavelength=cfg["wavelength"],
                    npix_image=H
                )
                
                variance = torch.tensor([1e-3], dtype=torch.float32, device=device)
                lengths = torch.tensor([images_2ch.shape[1]], dtype=torch.int64, device=device)

                coeff, num, den, psf, otf, train_loss = model(images_2ch, images_ft, variance, lengths=lengths)
                
                scaled_loss = (train_loss * loss_scale) / accumulation_steps
                scaled_loss.backward()
                
                train_epoch_losses.append(train_loss.item())

                if (step % accumulation_steps == 0) or (step == len(train_samples_shuffled)):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            train_loss_val = float(np.mean(train_epoch_losses))
            train_loss_history.append(train_loss_val)

            # Update learning rate at epoch end
            scheduler.step()

            # --- VALIDATION PHASE (1 shortest stack per telescope, deterministic frames 0..49, unaugmented) ---
            model.eval()
            val_epoch_losses = []
            with torch.no_grad():
                for sample_info in dataset.val_samples:
                    # Deterministic evaluation using first 50 frames (start_idx=0)
                    val_sample = dataset.sample_slice(sample_info, n_frames=n_frames_per_epoch, start_idx=0)
                    val_images_2ch = val_sample["images"].unsqueeze(0).to(device)
                    val_images_ft = val_sample["images_ft"].unsqueeze(0).to(device)
                    val_cfg = val_sample["config"]
                    val_H = val_images_2ch.shape[-2]

                    model.update_telescope_basis(
                        pixel_size=val_cfg["pixel_size"],
                        telescope_diameter=val_cfg["telescope_diameter"],
                        central_obscuration=val_cfg.get("central_obscuration", 0.0),
                        wavelength=val_cfg["wavelength"],
                        npix_image=val_H
                    )

                    _, _, _, _, _, val_loss = model(val_images_2ch, val_images_ft, variance, lengths=lengths)
                    val_epoch_losses.append(val_loss.item())

            val_loss_val = float(np.mean(val_epoch_losses))
            val_loss_history.append(val_loss_val)

            print(f"Epoch {epoch:03d}/{num_epochs} | LR: {scheduler.get_last_lr()[0]:.2e} | Train Loss: {train_loss_val:.6e} | Val Loss: {val_loss_val:.6e}")

            # --- CHECKPOINTING ---
            if val_loss_val < best_val_loss:
                best_val_loss = val_loss_val
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
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

        # Gráfica combinada (Entrenamiento y Validación juntas)
        plt.figure(figsize=(9, 5))
        plt.plot(train_loss_history, label="Training Loss", color="#1f77b4", linewidth=2)
        plt.plot(val_loss_history, label="Validation Loss", color="#ff7f0e", linewidth=2)
        plt.gca().ticklabel_format(useOffset=False, style='plain')
        plt.xlabel("Epoch")
        plt.ylabel("MOMFBD Loss")
        plt.title("Dynamic 50-Frame Stack Training & Validation Loss")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / "loss_plot.png", dpi=300)
        plt.close()

        # Gráfica individual de Entrenamiento
        plt.figure(figsize=(9, 5))
        plt.plot(train_loss_history, label="Training Loss", color="#1f77b4", linewidth=2)
        plt.gca().ticklabel_format(useOffset=False, style='plain')
        plt.xlabel("Epoch")
        plt.ylabel("MOMFBD Loss")
        plt.title("Training Loss")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / "train_loss_plot.png", dpi=300)
        plt.close()

        # Gráfica individual de Validación
        plt.figure(figsize=(9, 5))
        plt.plot(val_loss_history, label="Validation Loss", color="#ff7f0e", linewidth=2)
        plt.gca().ticklabel_format(useOffset=False, style='plain')
        plt.xlabel("Epoch")
        plt.ylabel("MOMFBD Loss")
        plt.title("Validation Loss")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / "val_loss_plot.png", dpi=300)
        plt.close()

        print(f"Finished! Run outputs written to {save_dir.resolve()}")

    except Exception as e:
        print("\n=================== ERROR TRACEBACK ===================", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("=========================================================\n", file=sys.stderr)