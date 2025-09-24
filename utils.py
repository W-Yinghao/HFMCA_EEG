"""
spectral data augmentation
- chaoqi Oct. 29
"""
import time
import torch
import numpy as np
from scipy.signal import spectrogram
import pickle
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, periodogram
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
from data_aug import *
from data_aug import random_FT_phase
import scipy.io
import numpy as np
import mne
import scipy.io as sio
import os


# 数据增强方法
def scale_frequency_bands(features, scale_range=(0.9, 1.1)):
    """
    随机缩放频域特征的特定频带。
    """
    scale_factors = np.random.uniform(scale_range[0], scale_range[1], size=features.shape[-1])
    enhanced_features = features * scale_factors
    return enhanced_features


# 建议测试多个 frequency 和 amplitude 组合，如 (2, 0.05) 或 (5, 0.1)，以观察对性能的实际影响。
def add_periodic_perturbation(features, frequency=2, amplitude=0.05):
    """
    向频域特征添加正弦波扰动，增强周期性。
    """
    num_bands = features.shape[-1]
    time = np.linspace(0, 2 * np.pi, num_bands)
    sinusoid = amplitude * np.sin(frequency * time)
    enhanced_features = features + sinusoid
    return enhanced_features



# 通道名顺序 seed
ch_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1',
            'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
            'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
            'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1',
            'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
            'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
            'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']

def split_eeg_windows(eeg_data, sfreq=200, window_sec=1, step_sec=None, mode='drop'):
    """
    Parameters:
        eeg_data: np.array (channels, time)
        sfreq: sampling frequency (Hz)
        window_sec: window length in seconds
        step_sec: step length in seconds (for overlap); default = no overlap
        mode: 'drop' to ignore last partial, 'tail' to ensure last full window ends at end

    Returns:
        windows: np.array (num_windows, channels, window_size)
    """
    channels, total_timepoints = eeg_data.shape
    window_size = int(window_sec * sfreq)
    step_size = int(step_sec * sfreq) if step_sec else window_size

    windows = []
    starts = list(range(0, total_timepoints - window_size + 1, step_size))

    if mode == 'tail':
        last_start = total_timepoints - window_size
        if last_start not in starts:
            starts.append(last_start)

    for start in starts:
        window = eeg_data[:, start:start + window_size]
        windows.append(window)

    return np.stack(windows)

 
def read_one_file(file_path, sfreq=200, window_sec=1, mode="drop"):
    """
    input:单个.mat文件路径
    output:raw格式数据
    """
    # 每个.mat文件中的数据label
    basic_label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

    data = sio.loadmat(file_path)
    # 获取keys并转化为list，获取数据所在key
    keys = list(data.keys())[3:]
    # print(keys)
    # 获取数据
    raw_list = []
    label_list = []
    for i in range(len(keys)):
        # 获取数据
        stamp = data[keys[i]]
        # print(stamp.shape)
        # 创建info
        # info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        # 创建raw，取第5秒开始的数据
        # raw = mne.io.RawArray(stamp, info).crop(tmin=5)
        # 添加到raw_list
        
        split_raw = split_eeg_windows(stamp, sfreq=sfreq, window_sec=window_sec, step_sec=None, mode=mode)
        # print(split_raw.shape)
        # print(name33)
        len_x = split_raw.shape[0]
        split_raw_y = [basic_label[i]] * len_x
        # print(len(split_raw_y))
        # print(split_raw_y)
        # print(name33)
        raw_list.extend(split_raw)
        label_list.extend(split_raw_y)
    return raw_list, label_list
 
 
def read_all_files(path, test_subject, sfreq=200, window_sec=1, mode="drop", train=True):
    # 读取文件夹下所有.mat文件
    print("read_all_files start...")
    start_time = time.time()
    # 遍历Preprocessed_EEG文件夹下所有.mat文件
    data_list = []
    labels_list = []
    # 读取文件数量（每个文件中有15段数据）
    files_num = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.mat' and os.path.splitext(file)[0] != 'label':
                filename_without_ext = os.path.splitext(file)[0]
                
                # 提取下划线前的数字部分
                if '_' in filename_without_ext:
                    file_subject = filename_without_ext.split('_')[0]
                else:
                    file_subject = filename_without_ext  # 如果没有下划线，整个文件名就是subject
                
                if train and file_subject != str(test_subject):
                    file_path = os.path.join(root, file)
                    raw_list, label_list = read_one_file(file_path, sfreq=sfreq, window_sec=window_sec, mode=mode)
                    data_list.extend(raw_list)
                    labels_list.extend(label_list)
                    files_num += 1
                elif not train and file_subject == str(test_subject):
                    file_path = os.path.join(root, file)
                    raw_list, label_list = read_one_file(file_path, sfreq=sfreq, window_sec=window_sec, mode=mode)
                    data_list.extend(raw_list)
                    labels_list.extend(label_list)
                    files_num += 1
    
    # for root, dirs, files in os.walk(path):
    #     for file in files:
    #         if os.path.splitext(file)[1] == '.mat' and os.path.splitext(file)[0] != 'label':
    #             # print(os.path.splitext(file))
    #             if train and (not os.path.splitext(file)[0].startswith(test_subject)):
    #                 # print(file)
    #                 file_path = os.path.join(root, file)
    #                 raw_list, label_list = read_one_file(file_path, sfreq=sfreq, window_sec=window_sec, mode=mode)
    #                 # 将raw_list中的每一个元素添加到data_list
    #                 data_list.extend(raw_list)
    #                 labels_list.extend(label_list)
    #                 files_num += 1
    #             elif not train and (os.path.splitext(file)[0].startswith(test_subject)):
    #                 # print(file)
    #                 file_path = os.path.join(root, file)
    #                 raw_list, label_list = read_one_file(file_path, sfreq=sfreq, window_sec=window_sec, mode=mode)
    #                 # 将raw_list中的每一个元素添加到data_list
    #                 data_list.extend(raw_list)
    #                 labels_list.extend(label_list)
    #                 files_num += 1
                    
                # if files_num == max_files_num:
                #     break
 
    # 生成所有数据的label（每个文件中有15段数据，每段数据的label相同）
    # label_list = []
    # for i in range(int(files_num)):
    #     label_list.extend(basic_label)
    # 将label_list添加到data_list
    print("共读取了{}个文件".format(files_num))
    print("共有{}段数据".format(len(data_list)))
    # print("read ended using {}".format((time.time()-start_time)/60))
    return data_list, labels_list
 

# data_path = "/home/xuli/researchfiler/EEG_SSL/data/SEED/Preprocessed_EEG/"
# read_all_files(data_path, 46)


def denoise_channel(ts, bandpass, signal_freq, bound):
    """
    bandpass: (low, high)
    """
    nyquist_freq = 0.5 * signal_freq
    filter_order = 1
    
    low = bandpass[0] / nyquist_freq
    high = bandpass[1] / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    ts_out = lfilter(b, a, ts)

    ts_out[ts_out > bound] = bound
    ts_out[ts_out < -bound] = - bound

    return np.array(ts_out)

def noise_channel(ts, mode, degree, bound):
    """
    Add noise to ts
    
    mode: high, low, both
    degree: degree of noise, compared with range of ts    
    
    Input:
        ts: (n_length)
    Output:
        out_ts: (n_length)
        
    """
    len_ts = len(ts)
    num_range = np.ptp(ts)+1e-4 # add a small number for flat signal
    
    ### high frequency noise
    if mode == 'high':
        noise = degree * num_range * (2*np.random.rand(len_ts)-1)
        out_ts = ts + noise
        
    ### low frequency noise
    elif mode == 'low':
        noise = degree * num_range * (2*np.random.rand(len_ts//100)-1)
        x_old = np.linspace(0, 1, num=len_ts//100, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise, kind='linear')
        noise = f(x_new)
        out_ts = ts + noise
        
    ### both high frequency noise and low frequency noise
    elif mode == 'both':
        noise1 = degree * num_range * (2*np.random.rand(len_ts)-1)
        noise2 = degree * num_range * (2*np.random.rand(len_ts//100)-1)
        x_old = np.linspace(0, 1, num=len_ts//100, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise2, kind='linear')
        noise2 = f(x_new)
        out_ts = ts + noise1 + noise2

    else:
        out_ts = ts

    out_ts[out_ts > bound] = bound
    out_ts[out_ts < -bound] = - bound
        
    return out_ts



class SEEDLoader(torch.utils.data.Dataset):
    def __init__(self, dataset_name, test_subject="15", train=True, SS=True):
        # self.list_IDs = list_IDs
        # self.dir = dir
        self.SS = SS

        # self.label_list = ['W', 'R', 1, 2, 3]
        self.bandpass1 = (1, 5)
        self.bandpass2 = (30, 49)
        self.n_length = 100 * 30
        self.n_channels = 62
        self.n_classes = 3
        self.signal_freq = 200
        self.bound = 0.00025

        data_path = "/projects/EEG-foundation-model/SEED"
        data_list, label_list = read_all_files(data_path, test_subject, sfreq=200, window_sec=2, mode="tail", train=train)
        # if train:
        #    # shffule teh data
        self.data_list = np.array(data_list)
        self.label_list = np.array(label_list)
        print("data info")
        print("is train", train)
        print("test subject "+test_subject)
        print("data length ", len(data_list))
        print("label length ", len(label_list))


    def __len__(self):
        return len(self.data_list)

    def add_noise(self, x, ratio):
        """
        Add noise to multiple ts
        Input:
            x: (n_channel, n_length)
        Output:
            x: (n_channel, n_length)
        """
        for i in range(self.n_channels):
            if np.random.rand() > ratio:
                mode = np.random.choice(['high', 'low', 'both', 'no'])
                x[i,:] = noise_channel(x[i,:], mode=mode, degree=0.05, bound=self.bound)
        return x

    def remove_noise(self, x, ratio):
        """
        Remove noise from multiple ts
        Input:
            x: (n_channel, n_length)
        Output:
            x: (n_channel, n_length)
        """
        for i in range(self.n_channels):
            rand = np.random.rand()
            if rand > 0.75:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self.bound) +\
                        denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self.bound)
            elif rand > 0.5:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self.bound)
            elif rand > 0.25:
                x[i, :] = denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self.bound)
            else:
                pass
        return x

    def crop(self, x):
        l = np.random.randint(1, self.n_length - 1)
        x[:, :l], x[:, l:] = x[:, -l:], x[:, :-l]

        return x

    def augment(self, x):
        t = np.random.rand()
        if t > 0.75:
            x = permute_channels(x, chan2shuf = 6)
        elif t > 0.5:
            x = masking(x, masked_ratio=0.3)
        elif t > 0.25:
            x = channel_dropout(x)
        else:
            x = crop_and_resize(x)
        return x

    def __getitem__(self, index):
        # path = self.dir + self.list_IDs[index]
        # sample = pickle.load(open(path, 'rb'))
        X, y = self.data_list[index], self.label_list[index]

        y = torch.LongTensor([y])
        # print(X.shape)
        if self.SS:
            aug1 = self.augment(X.copy())
            # print(aug1.shape)
            aug2 = self.augment(X.copy())
            # print(aug2.shape)
            return torch.FloatTensor(aug1.copy()), torch.FloatTensor(aug2.copy())
        else:
            return torch.FloatTensor(X), y



class SEEDLoaderH(torch.utils.data.Dataset):
    def __init__(self, dataset_name, test_subject="15", train=True, SS=True):
        # self.list_IDs = list_IDs
        # self.dir = dir
        self.SS = SS

        # self.label_list = ['W', 'R', 1, 2, 3]
        self.bandpass1 = (1, 5)
        self.bandpass2 = (30, 49)
        self.n_length = 100 * 30
        self.n_channels = 62
        self.n_classes = 3
        self.signal_freq = 200
        self.bound = 0.00025

        data_path = "/projects/EEG-foundation-model/SEED"
        data_list, label_list = read_all_files(data_path, test_subject, sfreq=200, window_sec=2, mode="tail", train=train)
        # if train:
        #    # shffule teh data
        self.data_list = data_list
        self.label_list = label_list

        print("data info")
        print("is train", train)
        print("test subject "+test_subject)
        print("data length ", len(data_list))

    def __len__(self):
        return len(self.data_list)

    def add_noise(self, x, ratio):
        """
        Add noise to multiple ts
        Input:
            x: (n_channel, n_length)
        Output:
            x: (n_channel, n_length)
        """
        for i in range(self.n_channels):
            if np.random.rand() > ratio:
                mode = np.random.choice(['high', 'low', 'both', 'no'])
                x[i,:] = noise_channel(x[i,:], mode=mode, degree=0.05, bound=self.bound)
        return x

    def remove_noise(self, x, ratio):
        """
        Remove noise from multiple ts
        Input: 
            x: (n_channel, n_length)
        Output: 
            x: (n_channel, n_length)
        """
        for i in range(self.n_channels):
            rand = np.random.rand()
            if rand > 0.75:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self.bound) +\
                        denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self.bound)
            elif rand > 0.5:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self.bound)
            elif rand > 0.25:
                x[i, :] = denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self.bound)
            else:
                pass
        return x

    def crop(self, x):
        l = np.random.randint(1, self.n_length - 1)
        x[:, :l], x[:, l:] = x[:, -l:], x[:, :-l]

        return x

    def augment(self, x, t=None):
        t = np.random.rand()
        if t > 0.75:
            x = self.add_noise(x, ratio=0.5)
        elif t > 0.5:
            x = self.remove_noise(x, ratio=0.5)
        elif t > 0.25:
            x = self.crop(x)
        else:
            x = x[[1,0],:]
        return x

        return x

    def __getitem__(self, index):
        # path = self.dir + self.list_IDs[index]
        # sample = pickle.load(open(path, 'rb'))
        X, y = self.data_list[index], self.label_list[index]

        # 

        y = torch.LongTensor([y])

        if self.SS:
            # original aug in codes
            # aug1 = self.augment(X.copy())
            # aug2 = self.augment(X.copy())

            x = np.array(X.copy())
            # x_len = len(x)
            # print(x.shape)
            eeg = permute_channels(x, chan2shuf = 6) # .reshape(1, x.shape[0],x.shape[1])

            eeg_1 = masking(x, masked_ratio=0.3)
            eeg_2 = channel_dropout(x)
            eeg_3 = crop_and_resize(x)
            eeg_4 = random_FT_phase(x,0.2)
            eeg_5 = random_slope_scale(x)
            eeg_6 = warp_signal(x)
            # print(eeg_3.shape)

            # eeg_flipping = flipping(eeg.copy())

            eeg_aug = np.concatenate([
                                    eeg.reshape(1, x.shape[0],x.shape[1]), 
                                    eeg_1.reshape(1, x.shape[0],x.shape[1]),
                                    eeg_2.reshape(1, x.shape[0],x.shape[1]),
                                    # eeg_flipping.reshape(1, x.shape[0],x.shape[1]),
                                    eeg_3.reshape(1, x.shape[0],x.shape[1])
                                    # eeg_4.reshape(1, x.shape[0],x.shape[1])
                                    # eeg_5.reshape(1, x.shape[0],x.shape[1])
                                    # eeg_6.reshape(1, x.shape[0],x.shape[1])
                                    ],
                                    axis=0)

            # eeg_jittering = jittering(eeg.copy(), mean=0, var=1)
            # eeg_scaling = scaling(eeg.copy(), ratio=1.03)
            # eeg_permutated = permutation(eeg.copy(), n_segments=5, pertub_mode="random", seg_mode="equal")
            # eeg_aug = np.concatenate([eeg, eeg_jittering, eeg_scaling, eeg_flipping, eeg_permutated], axis=0)

            # data = torch.Tensor(eeg_aug)

            return torch.FloatTensor(x), torch.FloatTensor(eeg_aug.copy())
        else:
            return torch.FloatTensor(X), y




def temporal_interpolation(x, desired_sequence_length, mode='nearest', use_avg=True):
    # print(x.shape)
    # squeeze and unsqueeze because these are done before batching
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)
    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    # Supports batch dimension
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")
  


def read_deap_by_mode_nolapping(mode, test_size=0.2):

    from sklearn.model_selection import train_test_split
    assert mode in ["arousal", "valence", "cls4"]

    if mode == "arousal":
        with open("DEAP_processed/deap_hala_x_noOverlapping_4s", "rb") as fp:
            x_list = pickle.load(fp)

        with open("DEAP_processed/deap_hala_y_noOverlapping_4s", "rb") as fp:
            y_list = pickle.load(fp)

    elif mode == "valence":

        with open("DEAP_processed/deap_hvlv_x_noOverlapping_4s", "rb") as fp:
            x_list = pickle.load(fp)

        with open("DEAP_processed/deap_hvlv_y_noOverlapping_4s", "rb") as fp:
            y_list = pickle.load(fp)
    else:
        with open("DEAP_processed/deap_av_x_noOverlapping_4s", "rb") as fp:
            x_list = pickle.load(fp)

        with open("DEAP_processed/deap_av_y_noOverlapping_4s", "rb") as fp:
            y_list = pickle.load(fp)        
        
    X_all = []
    Y_all = []

    for i in range(len(x_list)):  # for each subject
        x = np.array(x_list[i])  # e.g., (24, 59, 32, 256)
        y = np.array(y_list[i])  # e.g., (1416,)

        # reshape x
        x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # x.shape = (1416, 32, 256)

        X_all.extend(x)
        Y_all.extend(y)

    # train-test split of DEAP:
    x_train_DEAP, x_test_DEAP, y_train_DEAP, y_test_DEAP = train_test_split(np.array(X_all), np.array(Y_all), random_state=42, test_size=test_size)
    # print("Training data size: ", len(x_train_DEAP))
    # print("Testing data size: ", len(x_test_DEAP))

    print(np.array(x_train_DEAP).shape, np.array(x_test_DEAP).shape)

    return x_train_DEAP, x_test_DEAP, y_train_DEAP, y_test_DEAP


def read_deap_by_mode(mode,test_size=0.3):

    from sklearn.model_selection import train_test_split
    assert mode in ["arousal", "valence", "cls4"]

    if mode == "arousal":
        with open("DEAP_processed/deap_hala_x", "rb") as fp:
            x_list = pickle.load(fp)

        with open("DEAP_processed/deap_hala_y", "rb") as fp:
            y_list = pickle.load(fp)

    elif mode == "valence":

        with open("DEAP_processed/deap_hvlv_x", "rb") as fp:
            x_list = pickle.load(fp)

        with open("DEAP_processed/deap_hvlv_y", "rb") as fp:
            y_list = pickle.load(fp)
    else:
        with open("DEAP_processed/deap_av_x", "rb") as fp:
            x_list = pickle.load(fp)

        with open("DEAP_processed/deap_av_y", "rb") as fp:
            y_list = pickle.load(fp)        
        
    X_all = []
    Y_all = []

    for i in range(len(x_list)):  # for each subject
        x = np.array(x_list[i])  # e.g., (24, 59, 32, 256)
        y = np.array(y_list[i])  # e.g., (1416,)

        # reshape x
        x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # x.shape = (1416, 32, 256)

        X_all.extend(x)
        Y_all.extend(y)

    # train-test split of DEAP:
    x_train_DEAP, x_test_DEAP, y_train_DEAP, y_test_DEAP = train_test_split(np.array(X_all), np.array(Y_all), random_state=42, test_size=test_size)
    # print("Training data size: ", len(x_train_DEAP))
    # print("Testing data size: ", len(x_test_DEAP))

    print(np.array(x_train_DEAP).shape, np.array(x_test_DEAP).shape)

    return x_train_DEAP, x_test_DEAP, y_train_DEAP, y_test_DEAP


class DEAP_ss(Dataset):
    '''
    A class need to input the Dataloader in the pytorch.
    '''
    def __init__(self, feature, label, ss=True):
        super().__init__()

        self.x = feature
        self.y = label
        self.SS = ss
        # print("Is self-supervised way: ", ss)
        # if not ss:
        #     print(len(self.x))
        #     print(len(self.y))
        #     print(self.y[1])

    def __len__(self):
        return len(self.y)

    def augment(self, x):
        t = np.random.rand()
        if t > 0.75:
            x = permute_channels(x, chan2shuf = 6)
        elif t > 0.5:
            x = masking(x, masked_ratio=0.3)
        elif t > 0.25:
            x = channel_dropout(x)
        else:
            x = crop_and_resize(x)
        return x

    def __getitem__(self, index):

        if self.SS:
            X = self.x[index] # .numpy()
            # print(type(X))
            aug1 = self.augment(X.copy())
            # print(aug1.shape)
            aug2 = self.augment(X.copy())
            # print(aug2.shape)
            return torch.FloatTensor(aug1.copy()), torch.FloatTensor(aug2.copy())
        else:
            # print(index, self.x[index].shape, type(self.x[index]))
            # print(torch.FloatTensor(self.x[index]).shape, torch.FloatTensor(self.y[index]).shape)
            xi, yi = np.ascontiguousarray(self.x[index]), np.ascontiguousarray(self.y[index])
            xi = torch.from_numpy(xi).float()
            yi = torch.from_numpy(yi).float()
            return xi, yi


            

class DEAP_ssH(Dataset):
    '''
    A class need to input the Dataloader in the pytorch.
    '''
    def __init__(self,feature,label, ss=True):
        super().__init__()

        self.x = feature
        self.y = label
        self.SS = ss
        self.choose = []

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        # X, y = self.x[index], self.y[index]
        # print(type(X), type(y))

        if self.SS:

            x = np.array(self.x[index].copy())

            eeg = permute_channels(x, chan2shuf = 6) # .reshape(1, x.shape[0],x.shape[1])

            eeg_1 = masking(x, masked_ratio=0.3)
            eeg_2 = channel_dropout(x)
            eeg_3 = crop_and_resize(x)
            eeg_4 = random_FT_phase(x,0.2)
            eeg_5 = random_slope_scale(x)
            # eeg_6 = phase_swap(x)
            eeg_flipping = flipping(eeg.copy())
            eeg_jittering = jittering(eeg.copy(), mean=0, var=1)
            eeg_scaling = scaling(eeg.copy(), ratio=1.03)
            # 数据增强方法
            eeg_7 = scale_frequency_bands(eeg.copy(), scale_range=(0.9, 1.1))
            # 建议测试多个 frequency 和 amplitude 组合，如 (2, 0.05) 或 (5, 0.1)，以观察对性能的实际影响。
            eeg_8 = add_periodic_perturbation(eeg.copy(), frequency=2, amplitude=0.05)
            eeg_9 = permutation(eeg.copy(), n_segments=5, pertub_mode="random", seg_mode="equal")

            eeg_aug = np.concatenate([
                                    eeg.reshape(1, x.shape[0],x.shape[1]), 
                                    eeg_1.reshape(1, x.shape[0],x.shape[1]),
                                    eeg_2.reshape(1, x.shape[0],x.shape[1]),
                                    # eeg_flipping.reshape(1, x.shape[0],x.shape[1]),
                                    eeg_3.reshape(1, x.shape[0],x.shape[1]),
                                    # eeg_4.reshape(1, x.shape[0],x.shape[1]),
                                    # eeg_5.reshape(1, x.shape[0],x.shape[1]),
                                    # eeg_6.reshape(1, x.shape[0],x.shape[1]),
                                    # eeg_7.reshape(1, x.shape[0],x.shape[1]),
                                    # eeg_8.reshape(1, x.shape[0],x.shape[1]),
                                    # eeg_9.reshape(1, x.shape[0],x.shape[1]),
                                    # eeg_jittering.reshape(1, x.shape[0],x.shape[1]),
                                    # eeg_scaling.reshape(1, x.shape[0],x.shape[1]),
                                    # eeg_flipping.reshape(1, x.shape[0],x.shape[1]),
                                    ],
                                    axis=0)

            return torch.FloatTensor(x), torch.FloatTensor(eeg_aug)
        else:
            return torch.FloatTensor(self.x[index]), torch.FloatTensor(self.y[index])
    

class BCI4_2A_ss(Dataset):
    '''
    A class need to input the Dataloader in the pytorch.
    '''
    def __init__(self,feature,label,subject_id=None,ss=True):
        super().__init__()

        self.x = feature
        self.y = label
        self.s = subject_id
        self.SS = ss
        # print("Is self-supervised way: ", ss)
        # if not ss:
        #     print(len(self.x))
        #     print(len(self.y))
        #     print(self.y[1])

    def __len__(self):
        return len(self.y)

    def augment(self, x):
        t = np.random.rand()
        if t > 0.75:
            x = permute_channels(x, chan2shuf = 6)
        elif t > 0.5:
            x = masking(x, masked_ratio=0.3)
        elif t > 0.25:
            x = channel_dropout(x)
        else:
            x = crop_and_resize(x)
        return x

    def __getitem__(self, index):

        if self.SS:
            X = self.x[index].numpy()
            aug1 = self.augment(X.copy())
            # print(aug1.shape)
            aug2 = self.augment(X.copy())
            # print(aug2.shape)
            return torch.FloatTensor(aug1.copy()), torch.FloatTensor(aug2.copy())
        else:
            return self.x[index], self.y[index]

    # def __getitem__(self, index):
    #     return self.x[index], self.y[index]
    
    def get_num_class(self, num_class=[1,1,1,1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            label = self.y[i]
            label = int(label)
            if num_class[label]>0:
                num_class[label]-=1
                res[label].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y     
    
    def get_num_subject(self, num_class=[1,1,1,1,1,1,1,1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            s = self.s[i]
            s = int(s)
            if num_class[s]>0:
                num_class[s]-=1
                res[s].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y    


class BCI4_2A_ssH(Dataset):
    '''
    A class need to input the Dataloader in the pytorch.
    '''
    def __init__(self,feature,label,subject_id=None, ss=True):
        super().__init__()

        self.x = feature
        self.y = label
        self.s = subject_id
        self.SS = ss

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        X, y = self.x[index].numpy(), self.y[index]

        if self.SS:

            x = np.array(X.copy())

            eeg = permute_channels(x, chan2shuf = 6) # .reshape(1, x.shape[0],x.shape[1])

            eeg_1 = masking(x, masked_ratio=0.3)
            eeg_2 = channel_dropout(x)
            eeg_3 = crop_and_resize(x)
            # eeg_4 = random_FT_phase(x,0.2)
            # eeg_5 = random_slope_scale(x)
            # eeg_6 = phase_swap(x)
            # eeg_flipping = flipping(eeg.copy())
            # eeg_jittering = jittering(eeg.copy(), mean=0, var=1)
            # eeg_scaling = scaling(eeg.copy(), ratio=1.03)
            # eeg_permutated = permutation(eeg.copy(), n_segments=5, pertub_mode="random", seg_mode="equal")

            eeg_aug = np.concatenate([
                                    eeg.reshape(1, x.shape[0],x.shape[1]), 
                                    eeg_1.reshape(1, x.shape[0],x.shape[1]),
                                    eeg_2.reshape(1, x.shape[0],x.shape[1]),
                                    # eeg_flipping.reshape(1, x.shape[0],x.shape[1]),
                                    eeg_3.reshape(1, x.shape[0],x.shape[1]),
                                    # eeg_4.reshape(1, x.shape[0],x.shape[1]),
                                    # eeg_5.reshape(1, x.shape[0],x.shape[1])
                                    ],
                                    axis=0)

            return torch.FloatTensor(x), torch.FloatTensor(eeg_aug.copy())
        else:
            return torch.FloatTensor(X), y
    
    def get_num_class(self, num_class=[1,1,1,1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            label = self.y[i]
            label = int(label)
            if num_class[label]>0:
                num_class[label]-=1
                res[label].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y     
    
    def get_num_subject(self, num_class=[1,1,1,1,1,1,1,1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            s = self.s[i]
            s = int(s)
            if num_class[s]>0:
                num_class[s]-=1
                res[s].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y    



class BCI4_2B_ss(Dataset):
    '''
    A class need to input the Dataloader in the pytorch.
    '''
    def __init__(self,feature,label,subject_id=None,ss=True):
        super().__init__()

        self.x = feature
        self.y = label
        self.s = subject_id
        self.SS = ss
        # print("Is self-supervised way: ", ss)
        # if not ss:
        #     print(len(self.x))
        #     print(len(self.y))
        #     print(self.y[1])

    def __len__(self):
        return len(self.y)

    def augment(self, x):
        t = np.random.rand()
        if t > 0.75:
            x = permute_channels(x, chan2shuf = -1)
        elif t > 0.5:
            x = masking(x, masked_ratio=0.3)
        elif t > 0.25:
            x = channel_dropout(x)
        else:
            x = crop_and_resize(x)
        return x

    def __getitem__(self, index):

        if self.SS:
            X = self.x[index].numpy()
            aug1 = self.augment(X.copy())
            # print(aug1.shape)
            aug2 = self.augment(X.copy())
            # print(aug2.shape)
            return torch.FloatTensor(aug1.copy()), torch.FloatTensor(aug2.copy())
        else:
            return self.x[index], self.y[index]

    # def __getitem__(self, index):
    #     return self.x[index], self.y[index]
    
    def get_num_class(self, num_class=[1,1,1,1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            label = self.y[i]
            label = int(label)
            if num_class[label]>0:
                num_class[label]-=1
                res[label].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y     
    
    def get_num_subject(self, num_class=[1,1,1,1,1,1,1,1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            s = self.s[i]
            s = int(s)
            if num_class[s]>0:
                num_class[s]-=1
                res[s].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y    


class BCI4_2B_ssH(Dataset):
    '''
    A class need to input the Dataloader in the pytorch.
    '''
    def __init__(self,feature,label,subject_id=None, ss=True):
        super().__init__()

        self.x = feature
        self.y = label
        self.s = subject_id
        self.SS = ss

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        X, y = self.x[index].numpy(), self.y[index]

        if self.SS:

            x = np.array(X.copy())
            eeg = x
            # x_len = len(x)
            # print(x.shape)
            # x = self.add_noise(x, ratio=0.5)
            # x = self.remove_noise(x, ratio=0.5)
            # x = self.crop(x)
            # eeg = self.add_noise(x, ratio=0.5) # .reshape(1, x.shape[0],x.shape[1])

            eeg_1 = masking(eeg.copy(), masked_ratio=0.3)
            # eeg_2 = self.remove_noise(x, ratio=0.5)
            eeg_3 = crop_and_resize(eeg.copy())

            # eeg_4 = random_FT_phase(x,0.2)
            # eeg_5 = random_slope_scale(x)
            # eeg_6 = phase_swap(x)
            eeg_flipping = flipping(eeg.copy())
            eeg_jittering = jittering(eeg.copy(), mean=0, var=1)
            # eeg_scaling = scaling(eeg.copy(), ratio=1.03)
            # # 数据增强方法
            # eeg_7 = scale_frequency_bands(eeg.copy(), scale_range=(0.9, 1.1))
            # # 建议测试多个 frequency 和 amplitude 组合，如 (2, 0.05) 或 (5, 0.1)，以观察对性能的实际影响。
            # eeg_8 = add_periodic_perturbation(eeg.copy(), frequency=2, amplitude=0.05)
            # eeg_9 = permutation(eeg.copy(), n_segments=5, pertub_mode="random", seg_mode="equal")


            eeg_aug = np.concatenate([
                                    eeg_1.reshape(1, x.shape[0],x.shape[1]), 
                                    # eeg_1.reshape(1, x.shape[0],x.shape[1]), 
                                    eeg_jittering.reshape(1, x.shape[0],x.shape[1]), 
                                    eeg_flipping.reshape(1, x.shape[0],x.shape[1]), 
                                    eeg_3.reshape(1, x.shape[0],x.shape[1])], axis=0)

            return torch.FloatTensor(x), torch.FloatTensor(eeg_aug.copy())
        else:
            return torch.FloatTensor(X), y
    
    def get_num_class(self, num_class=[1,1,1,1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            label = self.y[i]
            label = int(label)
            if num_class[label]>0:
                num_class[label]-=1
                res[label].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y     
    
    def get_num_subject(self, num_class=[1,1,1,1,1,1,1,1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            s = self.s[i]
            s = int(s)
            if num_class[s]>0:
                num_class[s]-=1
                res[s].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y    



# 欧氏空间的对齐方式 其中x：NxCxS
def EA(x,new_R = None):
    # print(x.shape)
    '''
    The Eulidean space alignment approach for EEG data.

    Arg:
        x:The input data,shape of NxCxS
        new_R：The reference matrix.
    Return:
        The aligned data.
    '''
    
    xt = np.transpose(x,axes=(0,2,1))
    # print('xt shape:',xt.shape)
    E = np.matmul(x,xt)
    # print(E.shape)
    R = np.mean(E, axis=0)
    # print('R shape:',R.shape)

    R_mat = scipy.linalg.fractional_matrix_power(R,-0.5)
    new_x = np.einsum('n c s,r c -> n r s',x,R_mat)
    if new_R is None:
        return new_x

    new_x = np.einsum('n c s,r c -> n r s',new_x,scipy.linalg.fractional_matrix_power(new_R,0.5))
    
    return new_x


def train_validation_split(x,y,validation_size,seed = None):
    '''
    Split the training set into a new training set and a validation set
    @author: WenChao Liu
    '''
    if seed:
        np.random.seed(seed)
    label_unique = np.unique(y)
    validation_x = []
    validation_y = []
    train_x = []
    train_y = []
    for label in label_unique:
        index = (y==label)
        label_num = np.sum(index)
        print("class-{}:{}".format(label,label_num))
        class_data_x = x[index]
        class_data_y = y[index]
        rand_order = np.random.permutation(label_num)
        class_data_x,class_data_y = class_data_x[rand_order],class_data_y[rand_order]
        print(class_data_x.shape)
        validation_x.extend(class_data_x[:int(label_num*validation_size)].tolist())
        validation_y.extend(class_data_y[:int(label_num*validation_size)].tolist())
        train_x.extend(class_data_x[int(label_num*validation_size):].tolist())
        train_y.extend(class_data_y[int(label_num*validation_size):].tolist())
    
    validation_x = np.array(validation_x)
    validation_y = np.array(validation_y).reshape(-1)
    
    train_x = np.array(train_x)
    train_y = np.array(train_y).reshape(-1)
    
    print(train_x.shape,train_y.shape)
    print(validation_x.shape,validation_y.shape)
    return train_x,train_y,validation_x,validation_y


def few_shot_data(sub,data_path, class_number = 4,shot_number = 1):
    
    sub_path = os.path.join(data_path,'sub{}_train'.format(sub),'Data.mat')
    data = sio.loadmat(sub_path)
    x,y = data['x_data'],data['y_data'].reshape(-1)
    result_x = []
    result_y = []
    for i in range(class_number):
        label_index = (y == i)
        result_x.extend(x[label_index][:shot_number])
        result_y.extend([i]*shot_number)
        
    return np.array(result_x),np.array(result_y)    
    

def get_bci4_4a_data(sub,data_path,few_shot_number=1, is_few_EA = False, target_sample=-1, use_avg=True, use_channels=None, ss=True, hireac=False, need_val=False):
    
    target_session_1_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))
    target_session_2_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_session_1_path)
    session_2_data = sio.loadmat(target_session_2_path)
    R = None

    if is_few_EA is True:
        session_1_x = EA(session_1_data['x_data'],R)
    else:
        session_1_x = session_1_data['x_data']
        
    if is_few_EA is True:
        session_2_x = EA(session_2_data['x_data'],R)
    else:
        session_2_x = session_2_data['x_data']
    
    # -- debug for BCIC 2b
    test_x_1 = torch.FloatTensor(session_1_x)      
    test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)

    test_x_2 = torch.FloatTensor(session_2_x)      
    test_y_2 = torch.LongTensor(session_2_data['y_data']).reshape(-1)
    
    if target_sample>0:
        test_x_1 = temporal_interpolation(test_x_1, target_sample, use_avg=use_avg)
        test_x_2 = temporal_interpolation(test_x_2, target_sample, use_avg=use_avg)

    if use_channels is not None:
        test_dataset = BCI4_2A_ss(torch.cat([test_x_1,test_x_2],dim=0)[:,use_channels,:],torch.cat([test_y_1,test_y_2],dim=0),ss=False)
    else:
        test_dataset = BCI4_2A_ss(torch.cat([test_x_1,test_x_2],dim=0),torch.cat([test_y_1,test_y_2],dim=0),ss=False)

    source_train_x = []
    source_train_y = []
    source_train_s = []
    
    source_valid_x = []
    source_valid_y = []
    source_valid_s = []
    subject_id = 0

    for i in range(1,10):

        if i == sub:
            continue
        
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        train_data = sio.loadmat(train_path)
    
        test_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(i))
        test_data = sio.loadmat(test_path)

        if is_few_EA is True:
            session_1_x = EA(train_data['x_data'],R)
        else:
            session_1_x = train_data['x_data']

        session_1_y = train_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_1_x,session_1_y,test_size = 0.1, stratify = session_1_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)

        if not need_val:
            source_train_x.extend(valid_x)
            source_train_y.extend(valid_y)
            source_train_s.append(torch.ones((len(valid_y),))*subject_id)

        if is_few_EA is True:
            session_2_x = EA(test_data['x_data'],R)
        else:
            session_2_x = test_data['x_data']

        session_2_y = test_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_2_x,session_2_y,test_size = 0.1,stratify = session_2_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        # if not need_val:
        #     source_train_x.extend(valid_x)
        #     source_train_y.extend(valid_y)
        #     source_train_s.append(torch.ones((len(valid_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)
        subject_id+=1
    
    if need_val:
        source_train_x = torch.FloatTensor(np.array(source_train_x))
        source_train_y = torch.LongTensor(np.array(source_train_y))
        source_train_s = torch.cat(source_train_s, dim=0)

        source_valid_x = torch.FloatTensor(np.array(source_valid_x))
        source_valid_y = torch.LongTensor(np.array(source_valid_y))
        source_valid_s = torch.cat(source_valid_s, dim=0)
        
        if target_sample>0:
            source_train_x = temporal_interpolation(source_train_x, target_sample, use_avg=use_avg)
            source_valid_x = temporal_interpolation(source_valid_x, target_sample, use_avg=use_avg)
        

        if use_channels is not None:
            if hireac:
                train_dataset = BCI4_2A_ssH(source_train_x[:,use_channels,:],source_train_y,source_train_s, ss=True)
            else:
                train_dataset = BCI4_2A_ss(source_train_x[:,use_channels,:],source_train_y,source_train_s, ss=ss)
        else:
            if hireac:
                train_dataset = BCI4_2A_ssH(source_train_x,source_train_y,source_train_s, ss=True)
            else:
                train_dataset = BCI4_2A_ss(source_train_x,source_train_y,source_train_s, ss=ss)

        if use_channels is not None:
            if hireac:
                valid_datset = BCI4_2A_ssH(source_valid_x[:,use_channels,:],source_valid_y,source_valid_s, ss=True)
            else:
                valid_datset = BCI4_2A_ss(source_valid_x[:,use_channels,:],source_valid_y,source_valid_s, ss=ss)
        else:
            if hireac:
                valid_datset = BCI4_2A_ssH(source_valid_x,source_valid_y,source_valid_s, ss=True)
            else:
                valid_datset = BCI4_2A_ss(source_valid_x,source_valid_y,source_valid_s, ss=ss)

        return train_dataset,valid_datset,test_dataset
    else:

        source_train_x = torch.FloatTensor(np.array(source_train_x))
        source_train_y = torch.LongTensor(np.array(source_train_y))
        source_train_s = torch.cat(source_train_s, dim=0)

        
        if target_sample>0:
            source_train_x = temporal_interpolation(source_train_x, target_sample, use_avg=use_avg)
            
        if use_channels is not None:
            if hireac:
                train_dataset = BCI4_2A_ssH(source_train_x[:,use_channels,:],source_train_y,source_train_s, ss=True)
            else:
                train_dataset = BCI4_2A_ss(source_train_x[:,use_channels,:],source_train_y,source_train_s, ss=ss)
        else:
            if hireac:
                train_dataset = BCI4_2A_ssH(source_train_x,source_train_y,source_train_s, ss=True)
            else:
                train_dataset = BCI4_2A_ss(source_train_x,source_train_y,source_train_s, ss=ss)
        
        return train_dataset,test_dataset
    

def get_bci4_cross(sub, data_path, ss=True, hireac=False, twoA=True):
    
    # get test datasets
    target_session_1_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))
    target_session_2_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_session_1_path)
    session_2_data = sio.loadmat(target_session_2_path)


    session_1_x = session_1_data['x_data']
    session_2_x = session_2_data['x_data']
    
    # -- debug for BCIC 2b
    test_x_1 = torch.FloatTensor(session_1_x)      
    test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)

    test_x_2 = torch.FloatTensor(session_2_x)      
    test_y_2 = torch.LongTensor(session_2_data['y_data']).reshape(-1)

    if twoA:
        test_dataset = BCI4_2A_ss(torch.cat([test_x_1,test_x_2],dim=0),torch.cat([test_y_1,test_y_2],dim=0), ss=False)
    else:
        test_dataset = BCI4_2B_ss(torch.cat([test_x_1,test_x_2],dim=0),torch.cat([test_y_1,test_y_2],dim=0), ss=False)

    source_train_x = []
    source_train_y = []
    source_train_s = []
    subject_id = 0

    for i in range(1,10):

        if i == sub:
            continue
        
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        train_data = sio.loadmat(train_path)
    
        test_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(i))
        test_data = sio.loadmat(test_path)


        session_1_x = train_data['x_data']
        session_1_y = train_data['y_data'].reshape(-1)

        source_train_x.extend(session_1_x)
        source_train_y.extend(session_1_y)
        source_train_s.append(torch.ones((len(session_1_y),))*subject_id)


        session_2_x = test_data['x_data']
        session_2_y = test_data['y_data'].reshape(-1)

        source_train_x.extend(session_2_x)
        source_train_y.extend(session_2_y)
        source_train_s.append(torch.ones((len(session_2_y),))*subject_id)

        subject_id+=1

    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    source_train_s = torch.cat(source_train_s, dim=0)


    if hireac:
        if twoA:
            pretext_dataset = BCI4_2A_ssH(source_train_x,source_train_y,source_train_s, ss=True)
        else:
            pretext_dataset = BCI4_2B_ssH(source_train_x,source_train_y,source_train_s, ss=True)
    else:
        if twoA:
            pretext_dataset = BCI4_2A_ss(source_train_x,source_train_y,source_train_s, ss=ss)
        else:
            pretext_dataset = BCI4_2B_ss(source_train_x,source_train_y,source_train_s, ss=ss)
    if twoA:
        train_dataset = BCI4_2A_ss(source_train_x,source_train_y,source_train_s, ss=False)
    else:
        train_dataset = BCI4_2B_ss(source_train_x,source_train_y,source_train_s, ss=False)

    return pretext_dataset, train_dataset, test_dataset


def get_bci4_within(data_path, ss=True, hireac=False, twoA=True):

    source_train_x = []
    source_train_y = []
    source_train_s = []

    source_test_x = []
    source_test_y = []
    source_test_s = []

    subject_id = 0

    for i in range(1,10):
        
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        train_data = sio.loadmat(train_path)
    
        test_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(i))
        test_data = sio.loadmat(test_path)


        session_1_x = train_data['x_data']
        session_1_y = train_data['y_data'].reshape(-1)
        # print(session_1_x[0].shape)

        source_train_x.extend(session_1_x)
        source_train_y.extend(session_1_y)
        source_train_s.append(torch.ones((len(session_1_y),))*subject_id)


        session_2_x = test_data['x_data']
        session_2_y = test_data['y_data'].reshape(-1)

        source_test_x.extend(session_2_x)
        source_test_y.extend(session_2_y)
        source_test_s.append(torch.ones((len(session_2_y),))*subject_id)

        subject_id+=1

    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    source_train_s = torch.cat(source_train_s, dim=0)

    source_test_x = torch.FloatTensor(np.array(source_test_x))
    source_test_y = torch.LongTensor(np.array(source_test_y))
    source_test_s = torch.cat(source_test_s, dim=0)

    if hireac:
        if twoA:
            pretext_dataset = BCI4_2A_ssH(source_train_x,source_train_y,source_train_s, ss=True)
        else:
            pretext_dataset = BCI4_2B_ssH(source_train_x,source_train_y,source_train_s, ss=True)
    else:
        if twoA:
            pretext_dataset = BCI4_2A_ss(source_train_x,source_train_y,source_train_s, ss=ss)
        else:
            pretext_dataset = BCI4_2B_ss(source_train_x,source_train_y,source_train_s, ss=ss)
    if twoA:
        train_dataset = BCI4_2A_ss(source_train_x,source_train_y,source_train_s, ss=False)
    else:
        train_dataset = BCI4_2B_ss(source_train_x,source_train_y,source_train_s, ss=False)

    if twoA:
        test_dataset = BCI4_2A_ss(source_test_x,source_test_y,source_test_s, ss=False)
    else:
        test_dataset = BCI4_2B_ss(source_test_x,source_test_y,source_test_s, ss=False)
        
    return pretext_dataset, train_dataset, test_dataset
 

if __name__ == '__main__':

    data_path = "./BCIC_2b_0_38HZ/"
    train_dataset, test_dataset = get_bci_cross(1, data_path)
    print(len(train_dataset), len(test_dataset))

    train_dataset, test_dataset = get_bci_within(data_path)
    print(len(train_dataset), len(test_dataset))
