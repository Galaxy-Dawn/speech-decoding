import os
import pickle
import numpy as np
import hydra
from scipy.stats import t, sem


def xcorr(x, y=None, maxlags=None, scale='coeff'):
    x = np.asarray(x)
    if y is None:
        y = x
    else:
        y = np.asarray(y)

    # Remove the mean
    x = x - np.mean(x)
    y = y - np.mean(y)

    N = len(x)
    M = len(y)

    r = np.correlate(x, y, mode='full')
    lags = np.arange(-M + 1, N)

    if maxlags is not None:
        mask = (lags >= -maxlags) & (lags <= maxlags)
        r = r[mask]
        lags = lags[mask]

    if scale == 'biased':
        r = r / N
    elif scale == 'unbiased':
        r = r / (N - np.abs(lags))
    elif scale == 'coeff':
        r = r / np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))

    return r, lags


def compute_maxr_lag(data1, data2, maxlags, thres=0.1):
    # data1: [n, t]; data2: [n, t]
    assert data1.shape[0] == data2.shape[0]
    n = data1.shape[0]
    max_r_list, lag_list = [], []
    for i in range(n):
        r, lags = xcorr(data1[i], data2[i], maxlags=maxlags, scale='coeff')
        max_r, max_r_index = np.max(r), np.argmax(r)
        if max_r >= thres:
            max_r_list.append(max_r)
            lag_list.append(lags[max_r_index])

    return np.array(max_r_list), np.array(lag_list)


def describe_with_ci(data, method="nonparametric", ci=0.95):
    data = np.asarray(data, dtype=float)
    n = len(data)

    mean = np.mean(data)
    std = np.std(data, ddof=1)
    data_min = np.min(data)
    data_max = np.max(data)

    result = {
        "mean": mean,
        "std": std,
        "min": data_min,
        "max": data_max,
        "method": method
    }

    if method == "parametric":
        alpha = 1 - ci
        t_crit = t.ppf(1 - alpha/2, df=n-1)
        se = sem(data)
        ci_low = mean - t_crit * se
        ci_high = mean + t_crit * se

        result["ci"] = (ci_low, ci_high)
        return result

    elif method == "nonparametric":
        lower = (1 - ci) / 2
        upper = 1 - lower
        ci_low = np.percentile(data, lower * 100)
        ci_high = np.percentile(data, upper * 100)

        result["ci"] = (ci_low, ci_high)
        return result

    else:
        raise ValueError("method must be 'parametric' or 'nonparametric'")


def get_syllable_data(all_info, channels):
    data_by_syllable = {}
    for item in all_info:
        data = item['data'][channels, item['start']:item['start']+1000]
        if data.shape[-1] < 1000:
            padding = np.zeros((data.shape[0], 1000 - data.shape[-1]))
            data = np.concatenate([data, padding], axis=-1)
        data = data[np.newaxis]
        if item['syllable'] not in data_by_syllable:
            data_by_syllable[item['syllable']] = [data]
        else:
            data_by_syllable[item['syllable']].append(data)

    for syllable, data_list in data_by_syllable.items():
        data_by_syllable[syllable] = np.concatenate(data_list, axis=0).mean(axis=0)
    return data_by_syllable


@hydra.main(config_path="../conf/", config_name="analyze", version_base="1.2")
def speaking_listening_lag_corr_all(cfg):

    speaking_listening_shared_channels = {
        'S1' : {'initial': [], 'final': []},
        'S2' : {'initial': [82, 83, 84, 85, 88], 'final': []},
        'S3' : {'initial': [], 'final': []},
        'S4' : {'initial': [], 'final': [52]},
        'S5' : {'initial': [13, 14, 15, 76, 77, 78, 79, 80], 'final': [14, 15, 77, 78, 79, 80]},
        'S6' : {'initial': [117, 118, 122], 'final': [117, 118, 122]},
        'S7' : {'initial': [83, 84, 85, 92, 93, 94], 'final': [83, 84, 85, 92, 93, 94]},
        'S8' : {'initial': [175, 177], 'final': []},
        'S9' : {'initial': [], 'final': []},
        'S10': {'initial': [96, 97, 98], 'final': [97]},
        'S11': {'initial': [10, 38], 'final': [12]},
        'S12': {'initial': [52, 53, 54, 55, 56, 57, 58], 'final': [53, 54, 55, 56, 57, 58]}
    }

    all_corrs, all_lags = [], []
    for subject, share_channel_dict in speaking_listening_shared_channels.items():
        shared_channels = sorted(list(set(share_channel_dict['initial']) | set(share_channel_dict['final'])))
        if len(shared_channels) == 0:
            print(f'{subject} has no speaking-listening shared channels')
            continue
        speaking_all_info = pickle.load(open(os.path.join(cfg.dir.data_dir, subject, f'processed_data', cfg.response_latency.data_source, f'all_info_speaking.pkl'), 'rb'))
        listening_all_info = pickle.load(open(os.path.join(cfg.dir.data_dir, subject, f'processed_data', cfg.response_latency.data_source, f'all_info_listening.pkl'), 'rb'))
        speaking_data = get_syllable_data(speaking_all_info, shared_channels)
        listening_data = get_syllable_data(listening_all_info, shared_channels)
        for i, channel in enumerate(shared_channels):
            speaking_channel_data, listening_channel_data = [], []
            for syllable in speaking_data.keys():
                speaking_channel_data.append(speaking_data[syllable][i][np.newaxis])
                listening_channel_data.append(listening_data[syllable][i][np.newaxis])
            speaking_channel_data = np.concatenate(speaking_channel_data, axis=0)
            listening_channel_data = np.concatenate(listening_channel_data, axis=0)
            corrs, lags = compute_maxr_lag(speaking_channel_data, listening_channel_data, maxlags=cfg.response_latency.maxlag, thres=0.1)
            lags_desc = describe_with_ci(lags)
            if lags_desc['std'] < 100:
                all_corrs.append(list(corrs))
                all_lags.append(list(lags))
    return all_corrs, all_lags


if __name__ == '__main__':
    speaking_listening_lag_corr_all()








