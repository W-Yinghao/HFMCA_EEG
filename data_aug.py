# source from https://github.com/MedMaxLab/selfEEG
# source from https://github.com/DL4mHealth/Contrastive-Learning-in-Medical-Time-Series-Survey
# We extend our sincere gratitude for the previous open-source work.


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import math
import random
import scipy
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift, ifft
from scipy import interpolate, signal
import torch
import warnings
warnings.filterwarnings("ignore")


def torch_pchip(
    x: "1D Tensor",
    y: "ND Tensor",
    xv: "1D Tensor",
    save_memory: bool = True,
    new_y_max_numel: int = 4194304,
) -> torch.Tensor:
    """
    performs the pchip interpolation on the last dimension of the input tensor.

    This function is a pytorch adaptation of the scipy's pchip_interpolate [pchip]_
    . It performs sp-pchip interpolation (Shape Preserving Piecewise Cubic Hermite
    Interpolating Polynomial) on the last dimension of the y tensor.
    x is the original time grid and xv new virtual grid. So, the new values of y at
    time xv are given by the polynomials evaluated at the time grid x.

    This function is compatible with GPU devices.

    Parameters
    ----------
    x: 1D Tensor
        Tensor with the original time grid. Must be the same length as the last
        dimension of y.
    y: ND Tensor
        Tensor to interpolate. The last dimension must be the time dimension of the
        signals to interpolate.
    xv: 1D Tensor
        Tensor with the new virtual grid, i.e. the time points where to interpolate
    save_memory: bool, optional
        Whether to perform the interpolation on subsets of the y tensor by
        recursive function calls or not. Does not apply if y is a 1-D tensor.
        If set to False memory usage can greatly increase (for example with a
        128 MB tensor, the memory usage of the function is 1.2 GB), but it can
        speed up the process. However, this is not the case for all devices and
        performance may also decrease.

        Default = True
    new_y_max_numel: int, optional
        The number of elements which the tensor needs to surpass in order to make
        the function start doing recursive calls. It can be considered as an
        indicator of the maximum allowed memory usage since the lower the number,
        the lower the memory used.

        Default = 256*1024*16 (approximately 16s of recording of a 256 Channel
        EEG sampled at 1024 Hz).

    Returns
    -------
    new_y: torch.Tensor
        The pchip interpolated tensor.

    Note
    ----
    Some technical information and difference with other interpolation can be found
    here: https://blogs.mathworks.com/cleve/2012/07/16/splines-and-pchips/

    Note
    ----
    have a look also at the Scipy's documentation:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html
    Some parts of the code are inspired from:
    https://github.com/scipy/scipy/blob/v1.10.1/scipy/interpolate/_cubic.py#L157-L302

    References
    ----------
    .. [pchip] https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.pchip_interpolate.html

    Example
    -------
    >>> from scipy.interpolate import pchip_interpolate
    >>> import numpy as np
    >>> import selfeeg.utils
    >>> import torch
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))*500
    >>> xnp = x.numpy()
    >>> x_pchip = utils.torch_pchip(torch.arange(1024), x, torch.linspace(0,1023,475)).numpy()
    >>> xnp_pchip = pchip_interpolate(np.arange(1024),xnp, np.linspace(0,1023,475), axis=-1)
    >>> print(
    ...     np.isclose(x_pchip, xnp_pchip, rtol=1e-3,atol=0.5*1e-3).sum()==16*32*475
    ... ) # Should return True

    """

    if len(x.shape) != 1:
        raise ValueError(
            ["Expected 1D Tensor for x but received a ", str(len(x.shape)), "-D Tensor"]
        )
    if len(xv.shape) != 1:
        raise ValueError(
            ["Expected 1D Tensor for xv but received a ", str(len(xv.shape)), "-D Tensor"]
        )
    if x.shape[0] != y.shape[-1]:
        raise ValueError("x must have the same length than the last dimension of y")

    # Initialize the new interpolated tensor
    Ndim = len(y.shape)
    new_y = torch.empty((*y.shape[: (Ndim - 1)], xv.shape[0]), device=y.device)

    # If save_memory and the new Tensor size is huge, call recursively for
    # each element in the first dimension
    if save_memory:
        if Ndim > 1:
            if ((torch.numel(y) / y.shape[-1]) * xv.shape[0]) > new_y_max_numel:
                for i in range(new_y.shape[0]):
                    new_y[i] = torch_pchip(x, y[i], xv)
                return new_y

    # This is a common part for every channel
    if x.device.type == "mps" or xv.device.type == "mps":
        # torch bucketize is not already implemented in mps unfortunately
        # need to pass in cpu and return to mps. Note that this is very slow
        # like 500 times slower. But at least it doesn't throw an error
        bucket = torch.bucketize(xv.to(device="cpu"), x.to(device="cpu")) - 1
        bucket = bucket.to(device=x.device)
    else:
        bucket = torch.bucketize(xv, x) - 1
    bucket = torch.clamp(bucket, 0, x.shape[0] - 2)
    tv_minus = (xv - x[bucket]).unsqueeze(1)
    infer_tv = torch.cat(
        (tv_minus**3, tv_minus**2, tv_minus, torch.ones(tv_minus.shape, device=tv_minus.device)), 1
    )

    h = x[1:] - x[:-1]
    Delta = (y[..., 1:] - y[..., :-1]) / h
    k = torch.sign(Delta[..., :-1] * Delta[..., 1:]) > 0
    w1 = 2 * h[1:] + h[:-1]
    w2 = h[1:] + 2 * h[:-1]
    whmean = (w1 / Delta[..., :-1] + w2 / Delta[..., 1:]) / (w1 + w2)

    slope = torch.zeros(y.shape, device=y.device)
    slope[..., 1:-1][k] = whmean[k].reciprocal()

    slope[..., 0] = ((2 * h[0] + h[1]) * Delta[..., 0] - h[0] * Delta[..., 1]) / (h[0] + h[1])
    slope_cond = torch.sign(slope[..., 0]) != torch.sign(Delta[..., 0])
    slope[..., 0][slope_cond] = 0
    slope_cond = torch.logical_and(
        torch.sign(Delta[..., 0]) != torch.sign(Delta[..., 1]),
        torch.abs(slope[..., 0]) > torch.abs(3 * Delta[..., 0]),
    )
    slope[..., 0][slope_cond] = 3 * Delta[..., 0][slope_cond]

    slope[..., -1] = ((2 * h[-1] + h[-2]) * Delta[..., -1] - h[-1] * Delta[..., -2]) / (
        h[-1] + h[-2]
    )
    slope_cond = torch.sign(slope[..., -1]) != torch.sign(Delta[..., -1])
    slope[..., -1][slope_cond] = 0
    slope_cond = torch.logical_and(
        torch.sign(Delta[..., -1]) != torch.sign(Delta[..., -1]),
        torch.abs(slope[..., -1]) > torch.abs(3 * Delta[..., 1]),
    )
    slope[..., -1][slope_cond] = 3 * Delta[..., -1][slope_cond]

    t = (slope[..., :-1] + slope[..., 1:] - Delta - Delta) / h
    a = (t) / h
    b = (Delta - slope[..., :-1]) / h - t

    py_coef = torch.stack((a, b, slope[..., :-1], y[..., :-1]), -1)
    new_y = (py_coef[..., bucket, :] * infer_tv).sum(axis=-1)
    return new_y

def jittering(sample, mean=0, var=1):
    jittering_sample = sample + np.random.normal(mean, var ** 0.5, sample.shape)
    return jittering_sample


def scaling(sample, ratio=1.1):
    scaling_sample = sample * ratio
    return scaling_sample


def flipping(sample):
    flipping_sample = np.flip(sample)
    return flipping_sample


def permutation(x, n_segments=5, pertub_mode="random", seg_mode ="equal"):
    """This function is for univariate, equal segmentation, and random permuation. But it's easy to expand to more modes."""
    
    T = x.shape[0]
    
    sublength = int(T/n_segments)
    augmented = np.zeros_like(x)
    idx = np.random.permutation(n_segments) # random shuffling the order
    for i in range(n_segments):
        j = idx[i]
        augmented[i*sublength: (i+1)*sublength] = x[j*sublength: (j+1)*sublength]
    return  augmented 


def random_mask(x, pertub_ratio=0.5):
    mask = np.random.choice([0, 1], size=(sample.shape[0]), p=[pertub_ratio, (1-pertub_ratio)])
    return x*mask


def time_bind_masking(sample, pertub_ratio=0.1):
    start = np.random.choice(sample.shape[0]-10)
    end = start + 50
    time_band_masked_sample = sample.copy()
    time_band_masked_sample[start:end]=0
    return time_band_masked_sample

def time_random_masking(sample, pertub_ratio=0.1):
    time_random_masked_sample = random_mask(sample, pertub_ratio)
    return time_random_masked_sample
 
def frequency_band_masking(sample, fs, low, high, order):
    b, a = signal.butter(order, [low*2/fs, high*2/fs], 'bandpass')   
    freq_masked_sample = signal.filtfilt(b, a, sample)  
    return freq_masked_sample


def frequency_random_masking(sample):
    freq_spectrum = fft(sample)  
    masked_spectrum = random_mask(freq_spectrum)
    freq_random_masked_sample = ifft(masked_spectrum)
    return freq_random_masked_sample


def filtering(sample, fs, low, high, order, pattern='highpass'):
    if pattern == 'bandpass':
        b, a = signal.butter(order, [low*2/fs, high*2/fs], pattern)   
    elif pattern == 'lowpass':
        b, a = signal.butter(order, low*2/fs, pattern)   
    elif pattern == 'highpass':
        b, a = signal.butter(order,  high*2/fs, pattern) 
    freq_masked_sample = signal.filtfilt(b, a, sample)  
    return freq_masked_sample


def channel_dropout(x, Nchan: int = None, batch_equal: bool = True):

    """

    puts to 0 a given (or random) amount of channels of the ArrayLike object.

    Channels are selected randomly.

    Parameters

    ---------

    x: ArrayLike

        The input Tensor or Array.

        The last two dimensions must refer to the EEG recording

        (Channels x Samples).

    Nchan: int, optional

        Number of channels to drop. If not given, the number of channels is chosen

        at random in the interval [1, (Channel_total // 4) +1 ].

        Default = None

    batch_equal: bool, optional

        Whether to apply the same channel drop to all EEG records or not.

        Default = True

    Returns

    -------

    x: ArrayLike

        the augmented version of the input Tensor or Array.

    Example

    -------

    >>> import torch

    >>> import selfeeg.augmentation as aug

    >>> x = torch.ones(32,1024)*2 + torch.sin(torch.linspace(0, 8*torch.pi,1024))

    >>> xaug = aug.channel_dropout(x, 3)

    >>> print( (xaug[0:,10]==0).sum()==3) # should return True

    """

    Ndim = len(x.shape)

    x_drop = torch.clone(x) if isinstance(x, torch.Tensor) else np.copy(x)

    if batch_equal or Ndim < 3:

        if Nchan is None:

            Nchan = random.randint(1, (x.shape[-2] // 4) + 1)

        else:

            if Nchan > x.shape[-2]:

                raise ValueError(

                    "Nchan can't be higher than the actual number" " of channels in the given EEG"

                )

            else:

                Nchan = int(Nchan)

        drop_chan = np.random.permutation(x.shape[-2])[:Nchan]

        if isinstance(x, torch.Tensor):

            drop_chan = torch.from_numpy(drop_chan).to(device=x.device)

        x_drop[..., drop_chan, :] = 0

    else:

        for i in range(x.shape[0]):

            x_drop[i] = channel_dropout(x[i], Nchan=Nchan, batch_equal=batch_equal)

    return x_drop

def masking(
    x, mask_number: int = 1, masked_ratio: float = 0.1, batch_equal: bool = True
):
    """
    puts to zero random portions of the ArrayLike object.

    masking is applied along its last dimension.
    The function will apply the same masking operation to all
    Channels of the same EEG. The number of portions to mask and the
    overall masked ratio can be set as input.


    Parameters
    ----------
    x: ArrayLike
        The input Tensor or Array.
        The last two dimensions must refer to the EEG recording
        (Channels x Samples).
    mask_number: int, optional
        The number of masking blocks, i.e., how many portions of the signal
        the function must mask. It must be a positive integer.
        Note that the created portions will have random length, but the
        overall masked ratio will be the one given as input.

        Default = 1
    masked_ratio: float, optional
        The overall percentage of the signal to mask.
        It must be a scalar in range [0,1].

        Default = 0.1
    batch_equal: bool, optional
        Whether to apply the same masking to all elements in the batch or not.
        It does apply only if x has more than 2 dimensions.

        Default = True

    Returns
    -------
    x: ArrayLike
        the augmented version of the input Tensor or Array.

    Note
    ----
    `mask_number` and `mask_ratio` should be used to tune the number and width
    of masked blocks. For example, given masked_ratio = 0.50 and mask_number = 3,
    the number of samples put to 0 will be in total half the signal length with 3
    distinct blocks of consecutive zeroes of random length.

    Example
    -------
    >>> import torch
    >>> import selfeeg.augmentation as aug
    >>> x = torch.ones(16,32,1024)*2 + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> xaug = aug.masking(x, 3, 0.5)
    >>> print( torch.isclose(((xaug[0,0]==0).sum()/len(xaug[0,0])),
    ...                      torch.tensor([0.5]), rtol=1e-8,atol=1e-8) )
    >>> # should return True

    """

    if not (isinstance(mask_number, int)) or mask_number <= 0:
        raise ValueError("mask_number must be a positive integer")
    if masked_ratio <= 0 or masked_ratio >= 1:
        raise ValueError(
            "mask_ratio must be in range (0,1), " "i.e. all values between 0 and 1 excluded"
        )

    Ndim = len(x.shape)
    x_masked = np.copy(x) if isinstance(x, np.ndarray) else torch.clone(x)
    if Ndim < 3 or batch_equal:

        # IDENTIFY LENGTH OF MASKED PIECES
        sample2mask = int(masked_ratio * x.shape[-1])
        pieces = [0] * mask_number
        piece_sum = 0
        for i in range(mask_number - 1):

            left = sample2mask - piece_sum - (mask_number - i + 1)
            minval = max(1, int((left / (mask_number - i) + 1) * 0.75))
            maxval = min(left, int((left / (mask_number - i) + 1) * 1.25))
            pieces[i] = random.randint(minval, maxval)
            piece_sum += pieces[i]
        pieces[-1] = sample2mask - piece_sum

        # IDENTIFY POSITION OF MASKED PIECES
        maxspace = x.shape[-1] - sample2mask
        spaces = [0] * mask_number
        space_sum = 0
        for i in range(mask_number):
            left = maxspace - space_sum - (mask_number - i + 1)
            spaces[i] = random.randint(1, int(left / (mask_number - i) + 1))
            space_sum += spaces[i]

        # APPLYING MASKING
        cnt = 0
        for i in range(mask_number):
            cnt += spaces[i]
            x_masked[..., cnt : cnt + pieces[i]] = 0
            cnt += pieces[i]

    else:
        for i in range(x.shape[0]):
            x_masked[i] = masking(
                x[i], mask_number=mask_number, masked_ratio=masked_ratio, batch_equal=batch_equal
            )

    return x_masked

def crop_and_resize(
    x,
    segments: int = 10,
    N_cut: int = 1,
    batch_equal: bool = True,
):
    """
    crops some segments of the ArrayLike object.

    Function is applied along the last dimension of the input ArrayLike object and
    is resized to its original dimension. To do that, crop_and_resize:

        1. divides the last dimension of x into N segments
        2. selects at random a subset segments
        3. removes the selected segments from x
        4. creates a new cropped version of x
        5. resamples the new cropped version to the original dimension.
           For this part pchip interpolation with a uniform virtual grid is used.

    Parameters
    ----------
    x: ArrayLike
        the input Tensor or Array.
        The last two dimensions must refer to the EEG recording (Channels x Samples).
    segments : int, optional
        The number of segments to consider when dividing the last dimension of x.
        This is not the number of segments to cut, but the number of segments in
        which the signal is partitioned (a subset of these segments will be
        removed based on N_cut).

        Default = 10
    N_cut : int, optional
        The number of segments to cut, i.e. the number of segments to remove.

        Default = 1
    batch_equal: bool, optional
        Whether to apply the same crop to all EEG record or not.

        Default = True

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.

    Example
    -------
    >>> import torch
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> xaug = aug.crop_and_resize(x,32, 15)

    """
    if isinstance(x, np.ndarray):
        x_crop = np.empty_like(x)
    else:
        x_crop = torch.empty_like(x, device=x.device)
    Ndim = len(x.shape)
    if (batch_equal) or (Ndim < 3):

        segment_len = x.shape[-1] // segments
        if isinstance(x, np.ndarray):
            seg_to_rem = np.random.randint(0, segments, N_cut, dtype=int)
            idx_to_rem = np.concatenate(
                [np.arange(segment_len * seg, segment_len * (seg + 1)) for seg in seg_to_rem]
            )
            new_x = np.delete(x, idx_to_rem, axis=-1)
            x_crop = interpolate.pchip_interpolate(
                np.linspace(0, x.shape[-1] - 1, new_x.shape[-1]),
                new_x,
                np.linspace(0, x.shape[-1] - 1, x.shape[-1]),
                axis=-1,
            )
        else:

            seg_to_rem = torch.randperm(segments)[:N_cut]
            idx_to_rem = torch.cat(
                [torch.arange(segment_len * seg, segment_len * (seg + 1)) for seg in seg_to_rem]
            )

            # https://stackoverflow.com/questions/55110047/
            allidx = torch.arange(x.shape[-1])
            combined = torch.cat((allidx, idx_to_rem, idx_to_rem))
            uniques, counts = combined.unique(return_counts=True)
            difference = uniques[counts == 1]
            difference = difference.to(device=x.device)
            new_x = x[..., difference]
            x_crop = torch_pchip(
                torch.linspace(0, x.shape[-1] - 1, new_x.shape[-1], device=x.device),
                new_x,
                torch.linspace(0, x.shape[-1] - 1, x.shape[-1], device=x.device),
            )

    else:
        for i in range(x.shape[0]):
            x_crop[i] = crop_and_resize(x[i], segments, N_cut, batch_equal)

    return x_crop

def permute_channels(
    x,
    chan2shuf: int = 5,
    mode: str = "random",
    channel_map: list = None,
    chan_net: list[str] = "all",
    batch_equal: bool = True,
) :
    """
    permutes the ArrayLike object along the EEG channel dimension.

    Given an input x where the last two dimensions must be
    (EEG_channels x EEG_samples), permutation_channels shuffles
    all or a subset of the EEG's channels. Shuffles can be done randomly
    or using specific networks (based on resting state functional
    connectivity networks).

    Parameters
    ----------
    x: ArrayLike
        The input Tensor or Array.
        The last two dimensions must refer to the EEG recording
        (Channels x Samples). Thus, permutation is applied on the
        second to last dimension.
    chan2shuf: int, optional
        The number of channels to shuffle. It must be greater than 1.
        Only exception is -1, which can be given to permute all the channels.

        Default = -1
    mode: str, optional
        How to permute the channels. Can be:

            - 'random': shuffle channels at random
            - 'network': shuffle channels which belongs to the same network.
              A network is a subset of channels whose activity is
              (with a minumum degree) correlated between each other.
              This mode support only a subset of 61
              channels of the 10-10 system.

        Default = "random"
    channel_map: list[str], optional
        The EEG channel map. Must be a list of strings
        or a numpy array of dtype='<U4' with channel names as elements.
        Channel name must be defined with capital letters (e.g. 'P04', 'FC5').
        If None is left the following 61 channel map is initialized:

            - ['FP1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5',
              'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1',
              'P1', 'P3', 'P5', 'P7', 'PO7', 'PO3', 'O1', 'OZ', 'POZ', 'PZ',
              'CPZ', 'FPZ', 'FP2', 'AF8', 'AF4', 'AFZ', 'FZ', 'F2', 'F4', 'F6',
              'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCZ', 'CZ', 'C2',
              'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4',
              'CP2', 'P2', 'P4', 'P6', 'P8', 'PO8', 'PO4', 'O2']

        Default = None
    chan_net: str or list[str], optional
        The list of networks to use if network mode is selected.
        Must be a list of string or a single string.
        Supported networks are "DMN", "DAN", "VAN", "SMN", "VFN", "FPN".
        Use 'all' to select all networks. To get a list of
        the channel names per network call the
        ``get_eeg_network_channel_names`` function.

        Default = 'all'
    batch_equal: bool, optional
        whether to apply the same permutation to all EEG record or not.
        If True, permute_signal is called recursively in order
        to permute each EEG differently.

        Default = True

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.

    Warnings
    --------
    Using **chan2shuf = -1** and **mode = 'network'** can result in a lower number
    of channels permuted compared to the whole list of channels included in the
    networks. This is due to the implementation of the permutation block in
    network mode, which iteratively applies the channel permutation at each
    single network, sequentially excluding channels already permuted in previous
    steps (networks are not mutually exclusive, there are overlapping channels).
    At some point a network may remain with only one channel to permute, which
    cannot be permuted since it is alone. This applies only in contexts where
    the number of channel is near the number of the selected ones.

    See Also
    --------
    get_channel_map_and_networks : creates the channel map and networks arrays.
    get_eeg_channel_network_names : prints the channel network arrays.

    Example
    -------
    >>> import torch
    >>> import numpy as np
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(61,4) + torch.arange(61).reshape(61,1)
    >>> xaug = aug.permute_channels(x,10)
    >>> print( (x[:,0]!=xaug[:,0]).sum()) # should output 10
    >>> # Try with network mode and check if channels not in the selected networks were permuted
    >>> eeg1010, chan_net = aug.get_channel_map_and_networks(chan_net=["DMN","VFN"])
    >>> chan2per = np.union1d(chan_net[0], chan_net[1])
    >>> a=np.intersect1d(eeg1010, chan2per, return_indices=True)[1]
    >>> b=torch.from_numpy( np.delete(np.arange(61),a))
    >>> xaug2 = aug.permute_channels(x,50, mode='network', chan_net=["DMN","VFN"])
    >>> print( ((x[:,0]!=xaug2[:,0]).sum())==50) # should output True
    >>> print( ((x[b,0]==xaug2[b,0]).sum())==len(b) ) # should output True

    """
    Nchan = x.shape[-2]
    Ndim = len(x.shape)

    # Check if given input is ok
    if (chan2shuf == 1) or (chan2shuf == 0) or (chan2shuf > Nchan):
        msgLog = """
        chan2shuf must be bigger than 1 and smaller than the number of channels
        in the recorded EEG. \n Default value is -1, which means all EEG channels
        are shuffled
        """
        raise ValueError(msgLog)
    if Ndim == 1:
        msg = "the last two dimensions of x must be [channel]*[time window]"
        raise ValueError(msg)

    chan2shuf = x.shape[-2] if chan2shuf == -1 else chan2shuf
    x2 = np.copy(x) if isinstance(x, np.ndarray) else torch.clone(x)
    if (Ndim < 3) or (batch_equal):

        
        if mode.lower() == "random":
            if isinstance(x, np.ndarray):
                idx = np.arange(Nchan, dtype=int)
                np.random.shuffle(idx)
                idx = idx[:chan2shuf]
                idxor = np.sort(idx)
                if len(idx) > 1:
                    while len(np.where(idx == idxor)[0]) > 0:
                        np.random.shuffle(idx)
            else:
                idx = torch.randperm(Nchan)
                idx = idx[:chan2shuf]
                idxor, _ = torch.sort(idx)
                if len(idx) > 1:
                    while torch.sum(torch.eq(idx, idxor)) != 0:
                        idx = idx[torch.randperm(idx.shape[0])]
                if x.device.type != "cpu":
                    idx = idx.to(device=x.device)
                    idxor = idxor.to(device=x.device)

        # solve problems related to network not having enough channel to permute
        # compared to the desired
        if idxor.shape != idx.shape:
            idx = idx[: idxor.shape[-1]]
        # apply defined shuffle
        xtemp = x[..., idx, :]
        x2[..., idxor, :] = xtemp

    else:
        # call recursively for each dimension until last 2 are reached
        for i in range(x.shape[0]):
            x2[i] = permute_channels(
                x[i],
                chan2shuf=chan2shuf,
                mode=mode,
                channel_map=channel_map,
                chan_net=chan_net,
            )

    return x2

def phase_swap(x):
    """
    Apply the phase swap data augmentation to the ArrayLike object.

    The phase swap data augmentation consists in merging the amplitude
    and phase components of biosignals from different sources to help
    the model learn their coupling.
    Specifically, the amplitude and phase of two randomly selected EEG samples
    are extracted using the Fourier transform.
    New samples are then generated by applying the inverse Fourier transform,
    combining the amplitude from one sample with the phase from the other.
    See the following paper for more information [phaseswap]_.

    Parameters
    ----------
    x : ArrayLike
        A 3-dimensional torch tensor or numpy array.
        The last two dimensions must refer to the EEG (Channels x Samples).

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.

    Note
    ----
    `Phase swap` ignores the class of each sample.


    References
    ----------
    .. [phaseswap] Lemkhenter, Abdelhak, and Favaro, Paolo.
      "Boosting Generalization in Bio-signal Classification by
      Learning the Phase-Amplitude Coupling". DAGM GCPR (2020).

    """

    Ndim = len(x.shape)
    if Ndim != 3:
        raise ValueError("x must be a 3-dimensional array or tensor")

    N = x.shape[0]

    if isinstance(x, torch.Tensor):
        # Compute fft, module and phase
        xfft = torch.fft.fft(x)
        amplitude = xfft.abs()
        phase = xfft.angle()
        x_aug = torch.clone(xfft)

        # Random shuffle indeces
        idx_shuffle = torch.randperm(N).to(device=x.device)
        idx_shuffle_1 = idx_shuffle[: (N // 2)]
        idx_shuffle_2 = idx_shuffle[(N // 2) : (N // 2) * 2]

        # Apply phase swap
        x_aug[idx_shuffle_1] = amplitude[idx_shuffle_1] * torch.exp(1j * phase[idx_shuffle_2])
        x_aug[idx_shuffle_2] = amplitude[idx_shuffle_2] * torch.exp(1j * phase[idx_shuffle_1])

        # Reconstruct the signal
        x_aug = (torch.fft.ifft(x_aug)).real.to(device=x.device)

    else:

        xfft = np.fft.fft(x)
        amplitude = np.abs(xfft)
        phase = np.angle(xfft)
        x_aug = np.copy(xfft)

        # Random shuffle indeces
        idx_shuffle = np.random.permutation(N)
        idx_shuffle_1 = idx_shuffle[: (N // 2)]
        idx_shuffle_2 = idx_shuffle[(N // 2) : (N // 2) * 2]

        # Apply phase swap
        x_aug[idx_shuffle_1] = amplitude[idx_shuffle_1] * np.exp(1j * phase[idx_shuffle_2])
        x_aug[idx_shuffle_2] = amplitude[idx_shuffle_2] * np.exp(1j * phase[idx_shuffle_1])

        # Reconstruct the signal
        x_aug = (np.fft.ifft(x_aug)).real

    return x_aug


def random_slope_scale(
    x,
    min_scale: float = 0.9,
    max_scale: float = 1.2,
    batch_equal: bool = True,
    keep_memory: bool = False,
):
    """
    randomly scales the first derivative of x.

    Given the input `ArrayLike` object **x** where the last two dimensions
    refers to the EEG channels and samples (1D tensor are also accepted),
    random_slope_scale calculates the first derivatives of each EEG records,
    here simplified as the difference between two consecutive values of the last
    dimension, and rescale each of them with a random factor selected from a
    uniform distribution between min_scale and max_scale. This transformation
    is similar to adding a random noise, but with the constraint that the first
    derivatives must keep the same sign of the original EEG (e.g. if a value is
    bigger than the previous one, then this is also true in the transformed data).

    Parameters
    ----------
    x: ArrayLike
        The input Tensor or Array.
        The last two dimensions must refer to the EEG recording (Channels x Samples).
    min_scale: float, optional
        The minimum rescaling factor to be applied. It must be a value bigger than 0.

        Default = 0.9
    max_scale: float, optional
        The maximum rescaling factor to be applied. It must be a value bigger
        than min_scale.

        Default = 1.2
    batch_equal: bool, optional
        Whether to apply the same rescale to all EEGs in the batch or not.
        This apply only if x has more than 2 dimensions, i.e. more than 1 EEG.

        Default: True
    keep_memory: bool, optional
        Whether to keep memory of the previous changes in slope and accumulate
        them during the transformation or not. Basically, instead of using:

            ``x_hat(n)= x(n-1) + scaling*( x(n)-x(n-1) )``

        with n>1, x_hat transformed signal, x original signal,
        keep_memory apply the following:

            ``x_hat(n)= x_hat(n-1) + scaling*( x(n)-x(n-1) )``

        Keep in mind that this may completely change the range of values,
        as consecutive increases in the slopes may cause a strong vertical
        shift of the signal. If set to True, it is suggested to set the scaling
        factor in the range [0.8, 1.2]

        Default: False

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.

    Example
    -------
    >>> import torch
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> xaug = aug.random_slope_scale(x)
    >>> diff1=torch.abs(xaug[0,0,1:] - xaug[0,0,:-1])
    >>> diff2=torch.abs(x[0,0,1:] - x[0,0,:-1])
    >>> print(
    ...     torch.logical_or(diff1<=(diff2*1.2),
    ...     diff1>=(diff2*0.9)).sum()
    ... ) # should return 1023

    """

    Ndim = len(x.shape)
    if min_scale < 0:
        raise ValueError("minimum scaling factor can't be lower than 0")
    if max_scale <= min_scale:
        raise ValueError("maximum scaling factor can't be" " lower than minimum scaling factor")

    if batch_equal:
        if isinstance(x, np.ndarray):
            scale_factor = np.random.uniform(
                min_scale, max_scale, size=tuple(x.shape[-2:] + np.array([0, -1]))
            )
        else:
            scale_factor = torch.empty(x.shape[-2], x.shape[-1] - 1, device=x.device).uniform_(
                min_scale, max_scale
            )
    else:
        if isinstance(x, np.ndarray):
            scale_factor = np.random.uniform(min_scale, max_scale, x.shape)[..., 1:]
        else:
            scale_factor = torch.empty_like(x, device=x.device)[..., 1:].uniform_(
                min_scale, max_scale
            )

    x_diff = x[..., 1:] - x[..., :-1]
    x_diff_scaled = x_diff * scale_factor
    if isinstance(x, np.ndarray):
        x_new = np.empty_like(x)
    else:
        x_new = torch.empty_like(x, device=x.device)
    x_new[..., 0] = x[..., 0]
    if keep_memory:
        x_new[..., 1:] = (
            np.cumsum(x_diff_scaled, Ndim - 1)
            if isinstance(x, np.ndarray)
            else torch.cumsum(x_diff_scaled, Ndim - 1)
        )
    else:
        x_new[..., 1:] = x[..., :-1] + x_diff_scaled

    return x_new

def new_random_fft_phase_odd(n, to_torch_tensor: bool = False, device="cpu"):
    """
    Method for random_fft_phase with even length vector.
    See ``random_FT_phase`` help.

    :meta private:

    """
    if to_torch_tensor:
        random_phase = 2j * np.pi * torch.rand((n - 1) // 2)
        new_random_phase = torch.cat(
            (torch.tensor([0.0]), random_phase, -torch.flipud(random_phase))
        ).to(device=device)
    else:
        random_phase = 2j * np.pi * np.random.rand((n - 1) // 2)
        new_random_phase = np.concatenate([[0.0], random_phase, -random_phase[::-1]])
    return new_random_phase


def new_random_fft_phase_even(n, to_torch_tensor: bool = False, device="cpu"):
    """
    Method for random_fft_phase with even length vector.
    See ``random_FT_phase`` help.

    :meta private:

    """
    if to_torch_tensor:
        random_phase = 2j * np.pi * torch.rand(n // 2 - 1)
        new_random_phase = torch.cat(
            (torch.tensor([0.0]), random_phase, torch.tensor([0.0]), -torch.flipud(random_phase))
        ).to(device=device)
    else:
        random_phase = 2j * np.pi * np.random.rand(n // 2 - 1)
        new_random_phase = np.concatenate([[0.0], random_phase, [0.0], -random_phase[::-1]])
    return new_random_phase


def random_FT_phase(x, value: float = 1, batch_equal: bool = True):
    """
    randomizes the phase of all signals in the input ArrayLike object.

    For more info, see [ftphase1]_.

    Parameters
    ----------
    x: ArrayLike
        the input Tensor or Array.
        The last two dimensions must refer to the EEG recording
        (Channels x Samples).
    value: float, optional
        The magnitude of the phase perturbation. It must be a value between
        (0,1], which will be used to rescale the interval
        [0, 2* 'pi'] in [0, value * 2 * 'pi']

        Default = 1
    batch_equal: bool, optional
        Whether to apply the same perturbation on all signals or not.
        Note that all channels of the same records will be perturbed
        in the same way to preserve cross-channel correlations.

        Default = True

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.

    References
    ----------
    .. [ftphase1] Rommel, Cédric, et al. "Data augmentation for learning predictive
      models on EEG: a systematic comparison."
      Journal of Neural Engineering 19.6 (2022): 066020.

    Example
    -------
    >>> import torch
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> xaug = aug.random_FT_phase(x, 0.8)
    >>> # see https://dsp.stackexchange.com/questions/87343/
    >>> phase_shift = torch.arccos( 2*((x[0,0,0:512]*xaug[0,0,:512]).mean()) )
    >>> a=torch.sin(torch.linspace(0, 8*torch.pi,1024) + phase_shift)
    >>> if (a[0] - xaug[0,0,0]).abs()>0.1:
    ...     a=torch.sin(torch.linspace(0, 8*torch.pi,1024) - phase_shift)
    >>> print((a - xaug[0,0]).mean()<1e-3)

    plot the results (required matplotlib to be installed)

    >>> plt.plot(x[0,0])
    >>> plt.plot(xaug[0,0])
    >>> plt.plot(a)
    >>> plt.show()

    """
    if value <= 0 or value > 1:
        raise ValueError("value must be a float in range (0,1]")
    Ndim = len(x.shape)
    x_phase = torch.clone(x) if isinstance(x, torch.Tensor) else np.copy(x)
    if batch_equal or Ndim < 3:
        n = x.shape[-1]
        if isinstance(x, torch.Tensor):
            random_phase = (
                new_random_fft_phase_even(n, True, x.device)
                if n % 2 == 0
                else new_random_fft_phase_odd(n, True, x.device)
            )
            FT_coeff = torch.fft.fft(x)
            x_phase = torch.fft.ifft(FT_coeff * torch.exp(value * random_phase)).real
        else:
            if n % 2 == 0:
                random_phase = new_random_fft_phase_even(n)
            else:
                new_random_fft_phase_odd(n)
            FT_coeff = fft(x)
            x_phase = ifft(FT_coeff * np.exp(value * random_phase)).real
    else:
        for i in range(x_phase.shape[0]):
            x_phase[i] = random_FT_phase(x[i], value, batch_equal)
    return x_phase
def warp_signal(
    x,
    segments: int = 10,
    stretch_strength: float = 2.0,
    squeeze_strength: float = 0.5,
    batch_equal: bool = True,
):
    """
    stretches and squeezes portions of the ArrayLike object.

    The function is applied along the last dimension of the input ArrayLike object.
    To do that warp_signal:

        1. divides the last dimension of x into N segments
        2. selects at random a subset segments
        3. stretches those segments according to stretch_strength
        4. squeezes other segments according to squeeze_strength
        5. resamples x to the original dimension. For this part pchip
           interpolation with a uniform virtual grid is used

    Parameters
    ----------
    x: ArrayLike
        The input Tensor or Array. The last two dimensions must refer to the
        EEG recording (Channels x Samples).
    segments : int, optional
        The number of segments to consider when dividing the last dimension of x.

        Default = 10
    stretch_strength : float, optional
        The stretch power, i.e. a multiplication factor which determines the number
        of samples the stretched segment must have.

        Default = 2.
    squeeze_strength : float, optional
        The squeeze power. The same as stretch but for the segments to squeeze.

        Default = 0.5
    batch_equal: bool, optional
        Whether to apply the same warp to all records or not.

        Default = True

    Returns
    -------
    x: ArrayLike
        The augmented version of the input Tensor or Array.

    Example
    -------
    >>> import torch
    >>> import selfeeg.augmentation as aug
    >>> x = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*torch.pi,1024))
    >>> xaug = aug.warp_signal(x,20)

    """
    Ndim = len(x.shape)
    if isinstance(x, np.ndarray):
        x_warped_final = np.empty_like(x)
    else:
        x_warped_final = torch.empty_like(x, device=x.device)

    if batch_equal or Ndim < 3:

        # set segment do stretch or squeeze
        seglen = x.shape[-1] / segments
        seg_range = np.arange(segments)
        stretch = np.random.choice(seg_range, random.randint(1, segments // 2), replace=False)
        squeeze = np.setdiff1d(seg_range, stretch)

        # pre-allocate warped vector to avoid continuous stack call
        Lseg = np.zeros((segments, 2), dtype=int)
        Lseg[:, 0] = (seg_range * seglen).astype(int)
        Lseg[:, 1] = ((seg_range + 1) * seglen).astype(int)
        Lseg = Lseg[:, 1] - Lseg[:, 0]
        Lsegsum = np.zeros(segments + 1)
        Lsegsum[1:] = np.cumsum(Lseg)

        # iterate over segments and stretch or squeeze each segment, then allocate to x_warped
        new_tgrid = [None] * (segments)
        for i in range(segments):
            if i in stretch:
                new_piece_dim = int(np.ceil(Lseg[i] * stretch_strength))
            else:
                new_piece_dim = int(np.ceil(Lseg[i] * squeeze_strength))
            if isinstance(x, torch.Tensor):
                L = torch.arange(math.ceil(new_piece_dim))
            else:
                L = np.arange(math.ceil(new_piece_dim))
            L = L * (Lsegsum[i + 1] - Lsegsum[i]) / (len(L) - 1) + Lsegsum[i]
            new_tgrid[i] = L[:-1] if i < segments - 1 else L

        if isinstance(x, np.ndarray):
            new_tgrid = np.concatenate(new_tgrid)
            old_tgrid = np.arange(x.shape[-1])
            x_warped = interpolate.pchip_interpolate(old_tgrid, x, new_tgrid, axis=-1)
            new_tgrid = np.linspace(0, old_tgrid[-1], x_warped.shape[-1])
            x_warped_final = interpolate.pchip_interpolate(new_tgrid, x_warped, old_tgrid, axis=-1)
        else:
            device = x.device
            new_tgrid = torch.cat(new_tgrid).to(device=device)
            old_tgrid = torch.arange(x.shape[-1]).to(device=device)
            x_warped = torch_pchip(old_tgrid, x, new_tgrid)
            new_tgrid = torch.linspace(0, old_tgrid[-1], x_warped.shape[-1]).to(device=device)
            x_warped_final = torch_pchip(new_tgrid, x_warped, old_tgrid)

    else:
        # Recursively call until second to last dim is reached
        for i in range(x.shape[0]):
            x_warped_final[i] = warp_signal(
                x[i], segments, stretch_strength, squeeze_strength, batch_equal
            )

    return x_warped_final
