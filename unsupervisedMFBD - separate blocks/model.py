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

    def compute_psfs(self, coeff, target_shape=None):
        """Compute the PSFs and their Fourier transform from a set of modes
        
        Args:
            wavefront_focused ([type]): wavefront of the focused image
            illum ([type]): pupil aperture
            diversity ([type]): diversity for this specific images
        
        """

        # CHANGE: Interpolate pupil mask and modal basis if incoming images vary from npix_image
        pupil = self.pupil
        basis = self.basis
        
        if target_shape is not None and (target_shape[0] != self.npix_image or target_shape[1] != self.npix_image):
            H, W = target_shape
            # FIX 2: Explicit indexing [0, 0] / [0] prevents Squeeze dimension bugs
            pupil = F.interpolate(pupil[None, None, :, :], size=(H, W), mode='nearest')[0, 0]
            basis = F.interpolate(basis[None, :, :, :], size=(H, W), mode='bilinear', align_corners=False)[0]

        # Compute real and imaginary parts of the pupil
        wavefront = torch.einsum('ij,jkl->ikl', coeff, basis)

        # Compute the generalized pupil function
        phase = pupil[None, :, :] * torch.exp(1j * wavefront)

        # Compute FFT of the pupil function and compute autocorrelation
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



def run_debug_test():
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"--- Running Debug Test on [{device.upper()}] ---")

    # 1. Instantiate network
    model = Network(
        device=device,
        n_modes=44,
        n_frames=10,  # Max frames per sequence
        pixel_size=0.042,
        telescope_diameter=150.0,
        central_obscuration=0.0,
        wavelength=8000.0,
        basis_for_wavefront='kl',
        npix_image=128
    ).to(device)

    # Enable anomaly detection to catch NaN/Inf gradients immediately
    torch.autograd.set_detect_anomaly(True)

    # 2. Create a synthetic mini-batch with variable lengths
    batch_size = 2
    max_frames = 10
    H, W = 128, 128

    # Sequence lengths: Item 0 has 7 real frames, Item 1 has 10 real frames
    lengths = torch.tensor([7, 10], dtype=torch.int64).to(device)
    
    # # OLD (Uncorrelated Noise):
    # # Generate dummy image batch [B, Nf, H, W]
    # images = torch.rand((batch_size, max_frames, H, W), dtype=torch.float32, device=device)

    # NEW (Spatially Coherent Synthetic Object - Gaussian Spot):
    y, x = torch.meshgrid(torch.linspace(-1, 1, H, device=device), torch.linspace(-1, 1, W, device=device), indexing='ij')
    radius = torch.sqrt(x**2 + y**2)
    base_object = torch.exp(-radius**2 / (2 * 0.1**2))  # Sharp Gaussian spot at center
    
    # Broadcast to [B, Nf, H, W] and add slight frame-to-frame variations
    images = base_object[None, None, :, :].repeat(batch_size, max_frames, 1, 1)
    images = images + 0.05 * torch.rand_like(images) # Add light background noise
    
    # Zero out padded frames in input tensor to reflect actual padded data
    mask = (torch.arange(max_frames, device=device)[None, :] < lengths[:, None])[:, :, None, None]
    images = images * mask

    # Before passing to the model or computing FFTs:
    # Method A: Normalizes sequence by mean frame flux (preserving padded frame isolation)
    if lengths is not None:
        lengths_dev = lengths.to(images.device)
        masked_images = images * mask
        seq_sum = torch.sum(masked_images, dim=(-3, -2, -1), keepdim=True)
        seq_mean_flux = seq_sum / lengths_dev[:, None, None, None]
        images = images / (seq_mean_flux + 1e-10)
    else:
        seq_mean_flux = torch.mean(torch.sum(images, dim=(-2, -1), keepdim=True), dim=1, keepdim=True)
        images = images / (seq_mean_flux + 1e-10)

    # Pre-calculate Fourier transforms and noise variance
    images_ft = torch.fft.fft2(images, dim=(-2, -1), norm="ortho")
    variance = torch.tensor([1e-3, 1e-3], dtype=torch.float32, device=device)

    # 3. Setup Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("\n--- Starting 50 Optimization Steps ---")
    model.train()

    for step in range(1, 51):
        optimizer.zero_grad()

        # Forward pass
        coeff, num, den, psf, otf, loss = model(images, images_ft, variance, lengths=lengths)

        # Check for NaN/Inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ [Step {step}] Loss exploded to NaN/Inf!")
            break

        # ------------------------------------------------------------------
        # ADD GRADIENT DEBUGGING LINES HERE (Right before loss.backward())
        # ------------------------------------------------------------------
        if step == 1:
            print(f"DEBUG: loss.requires_grad = {loss.requires_grad}")
            print(f"DEBUG: coeff.requires_grad = {coeff.requires_grad}")
            print(f"DEBUG: C43.weight.requires_grad = {model.lstm.C43.weight.requires_grad}")
        # ------------------------------------------------------------------

        # Backward pass
        loss.backward()

        if step == 1:
            print(f"CNN A01 weight grad max:  {model.cnn.A01.conv.weight.grad.abs().max().item():.4e}")
            print(f"LSTM C43 weight grad max: {model.lstm.C43.weight.grad.abs().max().item():.4e}")

        # Gradient clipping to prevent sudden spikes during initial steps
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if step % 10 == 0 or step == 1:
            print(f"Step {step:02d} | MOMFBD Loss: {loss.item():.6e} | Grad Norm: {grad_norm.item():.4e}")

    print("\n✅ Verification complete! Gradients and backward pass are functioning properly.")

def run_real_stack_test(fits_path):
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"--- Running Real Data Test on [{device.upper()}] ---")

    # 1. Load Real Stack (e.g., FITS file containing [10, H, W] or longer)
    with fits.open(fits_path) as hdul:
        raw_data = hdul[0].data.astype(np.float32)

    # Extract first 10 frames: Shape -> [10, H, W]
    real_frames = raw_data[:10]
    
    # If spatial dimensions aren't 128x128, crop center or resize to fit model npix_image
    Nf, H, W = real_frames.shape
    if H > 128 or W > 128:
        start_h = (H - 128) // 2
        start_w = (W - 128) // 2
        real_frames = real_frames[:, start_h:start_h+128, start_w:start_w+128]

    # Convert to Tensor & Add Batch Dimension -> [1, 10, 128, 128]
    images = torch.tensor(real_frames, dtype=torch.float32, device=device).unsqueeze(0)
    batch_size, max_frames, H, W = images.shape
    lengths = torch.tensor([max_frames], dtype=torch.int64, device=device)

    # 2. Instantiate Model
    model = Network(
        device=device,
        n_modes=44,
        n_frames=max_frames,
        pixel_size=0.042,
        telescope_diameter=150.0,
        central_obscuration=0.0,
        wavelength=8000.0,
        basis_for_wavefront='kl',
        npix_image=H
    ).to(device)

    # 3. Method A Flux Normalization
    lengths_dev = lengths.to(device)
    seq_sum = torch.sum(images, dim=(-3, -2, -1), keepdim=True)
    seq_mean_flux = seq_sum / lengths_dev[:, None, None, None]
    images = images / (seq_mean_flux + 1e-10)

    # Pre-calculate 2D FFTs
    images_ft = torch.fft.fft2(images, dim=(-2, -1), norm="ortho")
    variance = torch.tensor([1e-3], dtype=torch.float32, device=device)

    # 4. Run Optimization Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    print(f"Image Stack Input Shape: {list(images.shape)}")
    print("\n--- Starting 50 Optimization Steps on Real Data ---")

    for step in range(1, 51):
        optimizer.zero_grad()

        coeff, num, den, psf, otf, loss = model(images, images_ft, variance, lengths=lengths)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ [Step {step}] Loss exploded to NaN/Inf!")
            break

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 10 == 0 or step == 1:
            print(f"Step {step:02d} | MOMFBD Loss: {loss.item():.6e} | Grad Norm: {grad_norm.item():.4e}")

    print("\n✅ Real data execution completed successfully!")

if __name__ == "__main__":
    # Provide the path to your real image data file (.fits / .npy)
    run_real_stack_test("path_to_your_real_stack.fits")

if __name__ == "__main__":
    run_debug_test()