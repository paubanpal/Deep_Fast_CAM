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

        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, bias=not bn)
        self.reflection = nn.ReflectionPad2d(kernel_size // 2)

        if self.use_bn:
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

        # Entrada de 1 canal: Imagen original normalizada
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
        if images.dim() == 5:
            B, Nf, C, H, W = images.shape
            tmp = images.view(B * Nf, C, H, W)
        elif images.dim() == 4:
            tmp = images
            B, Nf = images.shape[0], 1
        else:
            raise ValueError(f"Dimensión de entrada no soportada: {images.dim()}")

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

        if self.C43.bias is not None:
            nn.init.zeros_(self.C43.bias)

    def forward(self, latent_features, lengths=None):
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                latent_features, lengths.to(dtype=torch.int64, device='cpu'), batch_first=True, enforce_sorted=False
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
            
        self.pupil = None
        self.basis = None
        self.zeros = None
        self.basis_cache = {}

        self.cnn = CNN(n=16, n_lstm=256)
        self.cnn.weights_init()

        self.lstm = LSTM(n_modes=self.n_modes, n_lstm=256)
        self.lstm.weights_init()

    def update_telescope_basis(self, pixel_size, telescope_diameter, central_obscuration, wavelength, npix_image):
        if not hasattr(self, 'basis_cache'):
            self.basis_cache = {}

        if len(self.basis_cache) > 20:
            self.basis_cache.clear()
            torch.cuda.empty_cache()

        pixel_size_key = round(float(pixel_size), 6)
        config_key = (pixel_size_key, telescope_diameter, central_obscuration, wavelength, int(npix_image))
    
        if getattr(self, 'current_config', None) == config_key:
            return  

        self.pixel_size = pixel_size
        self.telescope_diameter = telescope_diameter
        self.central_obscuration = central_obscuration
        self.wavelength = wavelength
        self.npix_image = int(npix_image)
        self.current_config = config_key

        if config_key in self.basis_cache:
            cached_pupil, cached_basis, cached_zeros = self.basis_cache[config_key]
            self.pupil = cached_pupil.to(self.device)
            self.basis = cached_basis.to(self.device)
            self.zeros = cached_zeros.to(self.device)
            return

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

        self.basis_cache[config_key] = (pupil, basis_tensor, zeros)

        self.pupil = pupil
        self.basis = basis_tensor
        self.zeros = zeros

    def compute_psfs(self, coeff):
        wavefront = torch.einsum('ij,jkl->ikl', coeff, self.basis)
        phase = self.pupil[None, :, :] * torch.exp(1j * wavefront)

        ft = torch.fft.fft2(phase, norm="ortho")
        psf = (torch.conj(ft) * ft).real
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
    Carga stacks de imágenes originales directamente desde las subcarpetas de los telescopios.
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
                
            telescope_samples[tel_dir.name] = []

            for orig_file_path in sorted(tel_dir.glob("*.tif")):
                try:
                    with tiff.TiffFile(orig_file_path) as tf:
                        total_frames = len(tf.pages)
                except Exception as e:
                    print(f"[ERROR] Fallo al leer {orig_file_path.name}: {e}")
                    continue

                telescope_samples[tel_dir.name].append({
                    "orig_path": orig_file_path,
                    "config": tel_config,
                    "n_frames": total_frames,
                    "filename": orig_file_path.name,
                    "tel_dir_name": tel_dir.name
                })

        self.train_samples = []
        self.val_samples = []
        self.test_samples = []

        for tel_name, samples in telescope_samples.items():
            if len(samples) < 3:
                print(f"Warning: Telescopio {tel_name} tiene menos de 3 stacks válidos.")
                continue
                
            sorted_samples = sorted(samples, key=lambda s: s["n_frames"])
            
            self.val_samples.append(sorted_samples[0])
            self.test_samples.append(sorted_samples[1])
            self.train_samples.extend(sorted_samples[2:])

        print(f"Discovered {sum(len(s) for s in telescope_samples.values())} stack files across telescope subdirectories.")
        print(f"Split -> Train: {len(self.train_samples)} | Val: {len(self.val_samples)} | Test: {len(self.test_samples)}")

        if len(self.train_samples) == 0:
            raise RuntimeError("[ERROR CRÍTICO] La lista de muestras de entrenamiento está vacía.")

    def sample_slice(self, sample_info: dict, n_frames: int = 50, start_idx: int | None = None):
        """
        Extrae un slice contiguo de N fotogramas de la imagen original, 
        aplica la normalización y calcula su transformada de Fourier (FFT2).
        """
        orig_path: Path = sample_info["orig_path"]
        
        raw_orig = tiff.imread(orig_path).astype("float32")
        total_frames, H, W = raw_orig.shape

        if total_frames < n_frames:
            raise ValueError(f"Stack {orig_path.name} tiene solo {total_frames} fotogramas, pero se solicitaron {n_frames}.")

        max_start = total_frames - n_frames
        if start_idx is None:
            start_idx = random.randint(0, max_start)
        else:
            start_idx = min(start_idx, max_start)
        
        orig_slice = raw_orig[start_idx : start_idx + n_frames]
        orig_tensor = torch.tensor(orig_slice, dtype=torch.float32)

        # Normalización por el valor máximo del slice o stack
        orig_max = orig_tensor.amax(dim=(-2, -1), keepdim=True)
        orig_tensor = orig_tensor / (orig_max + 1e-8)

        # Aplicar recorte si es necesario
        if self.crop_dim is not None and (H > self.crop_dim or W > self.crop_dim):
            start_h = (H - self.crop_dim) // 2
            start_w = (W - self.crop_dim) // 2
            orig_tensor = orig_tensor[:, start_h:start_h + self.crop_dim, start_w:start_w + self.crop_dim]
            target_dim = self.crop_dim
        else:
            target_dim = H

        # Entrada de 1 canal para la CNN
        input_1ch = orig_tensor.unsqueeze(1) # [N_frames, 1, H, W]
        
        # Cálculo de la FFT2 compleja directamente sobre la imagen real normalizada
        fft_complex = torch.fft.fft2(orig_tensor, norm="ortho") # [N_frames, H, W]

        active_config = sample_info["config"].copy()
        active_config["target_dim"] = target_dim

        return {
            "images": input_1ch,             # [N_frames, 1, H, W]
            "images_ft": fft_complex,        # [N_frames, H, W] (complejo)
            "config": active_config,
            "filename": sample_info["filename"],
            "tel_dir_name": sample_info.get("tel_dir_name", ""),
            "start_frame": start_idx
        }

class AugmentedDatasetWrapper:
    def __init__(self, zoom_prob: float = 0.0, zoom_range: tuple = (1.05, 1.25)):
        self.zoom_prob = zoom_prob
        self.zoom_range = zoom_range

    def augment(self, sample: dict) -> dict:
        images = sample["images"]        
        images_ft = sample["images_ft"] 
        cfg = dict(sample["config"])
        
        N, C, H, W = images.shape
        
        angle = random.choice([0, 90, 180, 270])
        do_hflip = random.random() > 0.5
        do_vflip = random.random() > 0.5

        is_128 = (H == 128 and W == 128)
        zoom_factor = 1.0
        
        if is_128 and (random.random() < self.zoom_prob):
            zoom_factor = random.uniform(*self.zoom_range)

        augmented_frames = []
        for frame_1ch in images:
            if angle != 0:
                frame_1ch = TF.rotate(frame_1ch, angle)
            if do_hflip:
                frame_1ch = TF.hflip(frame_1ch)
            if do_vflip:
                frame_1ch = TF.vflip(frame_1ch)
                
            if is_128 and zoom_factor > 1.0:
                intermediate_H = int(H * zoom_factor)
                intermediate_W = int(W * zoom_factor)
                zoomed = F.interpolate(
                    frame_1ch.unsqueeze(0), size=(intermediate_H, intermediate_W), 
                    mode='bilinear', align_corners=False
                )
                frame_1ch = F.interpolate(
                    zoomed, size=(256, 256), mode='bilinear', align_corners=False
                ).squeeze(0)

            augmented_frames.append(frame_1ch)

        sample["images"] = torch.stack(augmented_frames, dim=0)
        
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

        if is_128 and zoom_factor > 1.0:
            total_scale_factor = 2.0 * zoom_factor
            cfg["pixel_size"] = cfg["pixel_size"] / total_scale_factor
            cfg["target_dim"] = 256

        sample["config"] = cfg
        return sample

def evaluate_reconstruction_and_modes(model_path, data_path, save_dir, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Usando dispositivo: {device}")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar Dataset de Validación
    print("[INFO] Cargando dataset de validación...")
    dataset = DynamicStackDataset(root_dir=data_path, seed=42)
    val_sample_info = dataset.val_samples[0]
    print(f"[INFO] Stack seleccionado: {val_sample_info['orig_path'].name}")
    
    val_sample = dataset.sample_slice(val_sample_info, n_frames=50, start_idx=0)
    images_1ch = val_sample["images"].unsqueeze(0).to(device)     # [1, 50, 1, H, W]
    images_ft = val_sample["images_ft"].unsqueeze(0).to(device)   # [1, 50, H, W]
    cfg = val_sample["config"]
    H, W = images_1ch.shape[-2], images_1ch.shape[-1]

    # 2. Inicializar y Cargar Modelo Entrenado
    print("[INFO] Cargando modelo y pesos...")
    model = Network(device=device, n_modes=119, n_frames=50, basis_for_wavefront='kl').to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    print("[INFO] Actualizando bases del telescopio...")
    model.update_telescope_basis(
        pixel_size=cfg["pixel_size"],
        telescope_diameter=cfg["telescope_diameter"],
        central_obscuration=cfg.get("central_obscuration", 0.0),
        wavelength=cfg["wavelength"],
        npix_image=H
    )

    # 3. Preprocesamiento e Inferencia
    print("[INFO] Ejecutando inferencia...")
    variance = torch.tensor([1e-3], dtype=torch.float32, device=device)
    lengths = torch.tensor([50], dtype=torch.int64, device=device)

    with torch.no_grad():
        coeff, num, den, psf, psf_ft, loss = model(images_1ch, images_ft, variance, lengths=lengths)

    # --- TAREA 1: Reconstrucción del Objeto (Filtro de Wiener) ---
    print("[INFO] Generando gráfica y archivos del objeto reconstruido...")
    eps = 1e-6
    object_ft = num / (den.real + variance[:, None, None] + eps)
    
    # 1. Aplicar ifftshift para mover la componente DC desde las esquinas al centro del dominio de Fourier
    object_ft_centered = torch.fft.ifftshift(object_ft, dim=(-2, -1))

    # 2. Inversa de Fourier a partir del espectro centrado
    object_spatial_complex = torch.fft.ifft2(object_ft_centered, norm="ortho")
    
    # (Opcional) Si quisieras asegurar el recentrado espacial absoluto, 
    # pero con ifftshift previo la estrella ya debería quedar en el centro exacto:
    object_reconstructed_raw = object_spatial_complex.real.squeeze().cpu().numpy()

    # Aplicar restricción física de no-negatividad
    object_reconstructed = np.clip(object_reconstructed_raw, a_min=0, a_max=None)
    
    # Guardar objeto reconstruido en TIFF y PNG
    obj_tiff_path = save_dir / "reconstructed_object.tiff"
    tiff.imwrite(obj_tiff_path, object_reconstructed.astype(np.float32))
    
    obj_single_png_path = save_dir / "reconstructed_object.png"
    plt.imsave(obj_single_png_path, object_reconstructed, cmap='gray')

    # Obtener la media de la entrada degradada directamente del sample de validación
    degraded_mean = images_1ch[0, :, 0, :, :].mean(dim=0).cpu().numpy()

    mean_tiff_path = save_dir / "degraded_mean_input.tiff"
    tiff.imwrite(mean_tiff_path, degraded_mean.astype(np.float32))
    
    mean_png_path = save_dir / "degraded_mean_input.png"
    plt.imsave(mean_png_path, degraded_mean, cmap='gray')

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(degraded_mean, cmap='gray')
    axes[0].set_title("Media Simple (Entrada Original)")
    axes[0].axis('off')

    axes[1].imshow(object_reconstructed, cmap='gray')
    axes[1].set_title("Objeto Reconstruido (Wiener)")
    axes[1].axis('off')

    plt.suptitle(f"Reconstrucción - Stack: {val_sample_info['orig_path'].name}", fontsize=14)
    plt.tight_layout()
    
    obj_plot_path = save_dir / "inspection_reconstructed_object.png"
    plt.savefig(obj_plot_path, dpi=300)
    plt.close()
    print(f"--> Gráfica comparativa guardada exitosamente en: {obj_plot_path}")

    # --- TAREA 2: Espectro de los Modos KL ---
    print("[INFO] Generando gráfica de decaimiento KL...")
    coeff_np = coeff.squeeze().cpu().numpy()
    mean_abs_coeff = np.mean(np.abs(coeff_np), axis=0)
    std_coeff = np.std(coeff_np, axis=0)

    mode_indices = np.arange(1, 120)

    plt.figure(figsize=(10, 5))
    plt.plot(mode_indices, mean_abs_coeff, marker='o', markersize=3, color='crimson', label=r'Amplitud Media $|\alpha_k|$')
    plt.fill_between(mode_indices, mean_abs_coeff - std_coeff, mean_abs_coeff + std_coeff, color='crimson', alpha=0.2, label=r'Desviación ($\sigma$)')
    
    plt.yscale('log')
    plt.xlabel("Índice del Modo KL")
    plt.ylabel("Amplitud del Coeficiente (rad)")
    plt.title("Espectro de Amplitud de los Modos KL")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    kl_plot_path = save_dir / "inspection_kl_modes_decay.png"
    plt.savefig(kl_plot_path, dpi=300)
    plt.close()
    print(f"--> Gráfica de modos KL guardada exitosamente en: {kl_plot_path}")

if __name__ == "__main__":
    import traceback
    try:
        model_ckpt = "/scratch/paulabp/TFM/run_outputs_v6_50_acc_sched_instance_norm_fixed/best_model.pt"
        data_dir = "/scratch/paulabp/TFM/images/images_for_network/originals_cropped"
        output_dir = "/scratch/paulabp/TFM/run_outputs_v6_50_acc_sched_instance_norm_fixed/run_outputs_comprobacion_50/plots"
        
        evaluate_reconstruction_and_modes(model_ckpt, data_dir, save_dir=output_dir)
    except Exception as e:
        print("\n=================== ERROR CAPTURADO ===================")
        traceback.print_exc()
        print("=======================================================\n")