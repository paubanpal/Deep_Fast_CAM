import torch
import numpy as np
import h5py
class Dataset(torch.utils.data.Dataset):
    """
    Dataset

      Scripts to produce the training sets : db.py
    
    """
    def __init__(self, filename, n_training_per_star=200, n_frames=10, validation=False):
        super(Dataset, self).__init__()
        
        # Read the video with the images
        self.filename = filename
        self.f = h5py.File(self.filename, 'r')
        self.datasets = [i for i in self.f.keys()]
        # if (not validation):
            # ind = [1, 13]
            # self.datasets = [self.datasets[i] for i in ind]
        self.n_training_per_star = n_training_per_star
        self.n_datasets = len(self.datasets)
        self.n_training = self.n_datasets * self.n_training_per_star
        self.n_frames = n_frames

        self.ind_time = []
        self.ind_dataset = []

        x, y = np.arange(128), np.arange(128)
        self.xx, self.yy = np.meshgrid(x, y)

        for dset in self.datasets:
            n, _ = self.f[dset].shape
            ind_time = np.random.randint(low=0, high=n-self.n_frames, size=self.n_training_per_star)
            self.ind_dataset.extend([dset] * self.n_training_per_star)
            self.ind_time.extend(ind_time)
        
        print(f"Number of training examples of {self.filename}: {self.n_training}")
                
    def __getitem__(self, index):
        dset = self.ind_dataset[index]
        low = self.ind_time[index]
        high = self.ind_time[index] + self.n_frames
        im = self.f[dset][low:high, :].reshape((self.n_frames, 128, 128))


        rot = np.random.randint(low=0, high=4, size=1)
        flipx = np.random.randint(low=0, high=2, size=1)
        flipy = np.random.randint(low=0, high=2, size=1)
        
        im = np.rot90(im, rot[0], axes=(1,2))
        if (flipx[0] == 1):
            im = im[:, ::-1, :]
        if (flipy[0] == 1):
            im = im[:, :, ::-1]

        max_im = np.max(im)
        min_im = np.min(im)
        
        im = (im - min_im) / (max_im - min_im)

        # im_aligned = np.zeros_like(im)
        # im_aligned[0, :, :] = im[0, :, :]
        # for i in range(self.n_frames-1):
        #     sh = align(im[i, :, :], im[i+1, :, :])
        #     im_aligned[i+1, :, :] = nd.interpolation.shift(im[i+1,:,:], sh, mode='wrap')

        # im = np.copy(im_aligned)

        # Make sure that the average is again at the center of the FOV
        tmp = np.sum(im, axis=0)

        delta = np.unravel_index(np.argmax(tmp), (128, 128))
        im = np.roll(im, (64-delta[0], 64-delta[1]), axis=(1, 2))

        ff = np.fft.fft2(im)
        im_fft = np.concatenate([ff.real[:, :, :, None], ff.imag[:, :, :, None]], axis=-1)

        variance = np.var(im[:, 0:10, 0:10])
        
        return im, im_fft, variance
        
    def __len__(self):
        return self.n_training