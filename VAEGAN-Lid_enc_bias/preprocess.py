import os
import time

from speech_tools import *

#dataset = 'vcc2018'

# speaker_list = ['SF1','SF2','SF3','SM1','SM2','TF1','TF2','TM1','TM2','TM3']
language_list = ['eng', 'kor']
# data_dir = os.path.join('data/speakers')
exp_dir = os.path.join('processed')
start_time = time.time()

sampling_rate = 16000
num_mcep = 36
frame_period = 5.0
n_frames = 128

for language in language_list:
# for speaker in speaker_list:

    if language == 'eng':
        num_samples = 8
        train_A_dir = os.path.join('..', 'english', 'training')
    elif language == 'kor':
        num_samples = 8
        train_A_dir = os.path.join('..', 'korean', 'training')

    # train_A_dir = os.path.join(data_dir, language)
    exp_A_dir = os.path.join(exp_dir, language)
    # train_A_dir = os.path.join(data_dir, speaker)
    # exp_A_dir = os.path.join(exp_dir, speaker)

    os.makedirs(exp_A_dir, exist_ok=True)
    #os.makedirs(exp_B_dir, exist_ok=True)
    print('Loading {} Wavs...'.format(language))
    wavs_A = load_wavs_random_sample(wav_dir = train_A_dir, sr = sampling_rate, num_samples = num_samples)
    # wavs_A = load_wavs(wav_dir=train_A_dir, sr=sampling_rate)
    #wavs_B = load_wavs(wav_dir=train_B_dir, sr=sampling_rate)

    print('Extracting acoustic features...')

    f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = world_encode_data(wavs=wavs_A, fs=sampling_rate,
                                                                    frame_period=frame_period, coded_dim=num_mcep)
    #f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = world_encode_data(wavs=wavs_B, fs=sampling_rate,
    #                                                                frame_period=frame_period, coded_dim=num_mcep)

    print('Calculating F0 statistics...')

    log_f0s_mean_A, log_f0s_std_A = logf0_statistics(f0s_A)
    #log_f0s_mean_B, log_f0s_std_B = logf0_statistics(f0s_B)

    # print('Log Pitch {}'.format(speaker))
    print('Log Pitch {}'.format(language))

    print('Mean: %f, Std: %f' % (log_f0s_mean_A, log_f0s_std_A))
    #print('Log Pitch B')
    #print('Mean: %f, Std: %f' % (log_f0s_mean_B, log_f0s_std_B))

    print('Normalizing data...')

    coded_sps_A_transposed = transpose_in_list(lst=coded_sps_A)
    #coded_sps_B_transposed = transpose_in_list(lst=coded_sps_B)

    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = coded_sps_normalization_fit_transoform(
        coded_sps=coded_sps_A_transposed)
    #coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = coded_sps_normalization_fit_transoform(
    #    coded_sps=coded_sps_B_transposed)

    print('Saving {} data...'.format(language))
    save_pickle(os.path.join(exp_A_dir, 'cache{}.p'.format(num_mcep)),
                (coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A))
    #save_pickle(os.path.join(exp_B_dir, 'cache{}.p'.format(num_mcep)),
    #            (coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std, log_f0s_mean_B, log_f0s_std_B))

print('Preprocess Complete.')

"""
train_A_dir = os.path.join(data_dir, speaker)
train_B_dir = os.path.join(data_dir, trg_speaker)
exp_A_dir = os.path.join(exp_dir, speaker)
exp_B_dir = os.path.join(exp_dir, trg_speaker)

os.makedirs(exp_A_dir, exist_ok=True)
os.makedirs(exp_B_dir, exist_ok=True)

sampling_rate = 22050
num_mcep = 36
frame_period = 5.0
n_frames = 128

print('Loading Wavs...')

start_time = time.time()

wavs_A = load_wavs(wav_dir=train_A_dir, sr=sampling_rate)
wavs_B = load_wavs(wav_dir=train_B_dir, sr=sampling_rate)

print('Extracting acoustic features...')

f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = world_encode_data(wavs=wavs_A, fs=sampling_rate,
                                                                 frame_period=frame_period, coded_dim=num_mcep)
f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = world_encode_data(wavs=wavs_B, fs=sampling_rate,
                                                                 frame_period=frame_period, coded_dim=num_mcep)

print('Calculating F0 statistics...')

log_f0s_mean_A, log_f0s_std_A = logf0_statistics(f0s_A)
log_f0s_mean_B, log_f0s_std_B = logf0_statistics(f0s_B)

print('Log Pitch A')
print('Mean: %f, Std: %f' % (log_f0s_mean_A, log_f0s_std_A))
print('Log Pitch B')
print('Mean: %f, Std: %f' % (log_f0s_mean_B, log_f0s_std_B))

print('Normalizing data...')

coded_sps_A_transposed = transpose_in_list(lst=coded_sps_A)
coded_sps_B_transposed = transpose_in_list(lst=coded_sps_B)

coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = coded_sps_normalization_fit_transoform(
    coded_sps=coded_sps_A_transposed)
coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = coded_sps_normalization_fit_transoform(
    coded_sps=coded_sps_B_transposed)

print('Saving data...')
save_pickle(os.path.join(exp_A_dir, 'cache{}.p'.format(num_mcep)),
            (coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A))
save_pickle(os.path.join(exp_B_dir, 'cache{}.p'.format(num_mcep)),
            (coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std, log_f0s_mean_B, log_f0s_std_B))
"""
end_time = time.time()
time_elapsed = end_time - start_time

print('Preprocessing Done.')

print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (
    time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))
