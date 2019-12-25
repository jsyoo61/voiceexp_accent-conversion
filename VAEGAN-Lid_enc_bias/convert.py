import json
import os
from utility import *
import tensorflow as tf
import numpy as np
import soundfile as sf
#from preprocess import *
from util.wrapper import load
# from analyzer import read_whole_features, pw2wav
# from analyzer import Tanhize
from datetime import datetime
from importlib import import_module
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from model.vawgan_multi_decoder import encode, decode,discriminate
from util.layers import GaussianLogDensity, GaussianKLD, \
    GaussianSampleLayer, lrelu
from speech_tools import *
import argparse
import soundfile
all_speaker = get_speakers(trainset='./data/speakers')
label_enc = LabelEncoder()
label_enc.fit(all_speaker)

#dataset = 'vcc2018'
def conversion(model_dir, source, target):
    src_speaker = source
    trg_speaker = target
    #model_name = 'cyclegan_vc2'

    data_dir = os.path.join('data')
    exp_dir = os.path.join('processed')

    if source == 'eng':
        converting_data_dir = os.path.join('..', 'english', 'test')
    elif source == 'kor':
        converting_data_dir = os.path.join('..', 'korean', 'test')
    # eval_B_dir = os.path.join(data_dir, 'speakers_test', trg_speaker)
    exp_A_dir = os.path.join(exp_dir, src_speaker)
    exp_B_dir = os.path.join(exp_dir, trg_speaker)

    validation_A_output_dir = os.path.join('converted_voices', 'converted_{}_to_{}'.format(src_speaker, trg_speaker))
    #validation_B_output_dir = os.path.join('converted_voices', dataset, model_name,
    #                                       'converted_{}_to_{}'.format(trg_speaker, src_speaker))

    os.makedirs(validation_A_output_dir, exist_ok=True)
    #os.makedirs(validation_B_output_dir, exist_ok=True)

    sampling_rate = 16000
    num_mcep = 36
    frame_period = 5.0
    n_frames = 128
    num_samples = 2

    print('Loading cached data...')
    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A = load_pickle(
        os.path.join(exp_A_dir, 'cache{}.p'.format(num_mcep)))
    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std, log_f0s_mean_B, log_f0s_std_B = load_pickle(
        os.path.join(exp_B_dir, 'cache{}.p'.format(num_mcep)))


    speaker_list = ['SF1','SF2','SF3','SM1','SM2','TF1','TF2','TM1','TM2','TM3']
    language_list = ['eng', 'kor']
    num_speaker = len(speaker_list)
    num_language = len(language_list)
    src_num = language_list.index(source)
    trg_num = language_list.index(target)

    print("source num ",src_num)
    print("target num ",trg_num)

    #input_shape = [None, num_mcep, None, 1]
    input_shape = [None, num_mcep, None]
    x_test = tf.placeholder(tf.float32, input_shape, name='test_input')
    is_training=False
    #machine = MODEL(arch)
    #y_t=1
    y_A = tf.placeholder(tf.float32, shape = [num_language], name = 'A_lang_vector')
    y_B = tf.placeholder(tf.float32, shape = [num_language], name = 'B_lang_vector')
    # y_A = tf.placeholder(tf.float32, shape = [10], name = 'id_vector_A')
    # y_B = tf.placeholder(tf.float32, shape = [10], name = 'id_vector_B')

    z_mu, z_lv = encode(x_test, y_A,is_training=is_training,scope_name = 'encoder')
    #z = GaussianSampleLayer(z_mu, z_lv)
    x_t = decode(z_mu, y_B,is_training=is_training,scope_name='decoder',mode='test')  # NOTE: the API yields NHWC format
    saver = tf.train.Saver()

    # Create one-hot vector
    A_id = [0.] * num_language
    B_id = [0.] * num_language

    A_id[src_num] = 1.0
    B_id[trg_num] = 1.0

    # Get filepaths of files to convert
    # flist = sorted(glob.glob(converting_data_dir + '/*.wav'))
    flist = get_file_paths_random_sample(dir = converting_data_dir, num_samples = num_samples)

    with tf.Session() as sess:
        saver.restore(sess, model_dir)

        for file in flist:
                print(file)
                wav, _ = librosa.load(file, sr=sampling_rate, mono=True)
                #wav = librosa.util.normalize(wav, norm=np.inf, axis=None)
                wav = wav_padding(wav=wav, sr=sampling_rate, frame_period=frame_period, multiple=4)
                f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
                f0_converted = pitch_conversion(f0=f0, mean_log_src=log_f0s_mean_A, std_log_src=log_f0s_std_A,
                                                mean_log_target=log_f0s_mean_B, std_log_target=log_f0s_std_B)
                coded_sp = world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
                #leng_sp = len(coded_sp.T)

                coded_sp_transposed = coded_sp.T

                coded_sp_norm = (coded_sp_transposed - coded_sps_A_mean) / coded_sps_A_std
                #print(len(coded_sps_A_mean))
                    #coded_sp_converted_norm = model.test(inputs=np.array([coded_sp_norm]), direction='A2B')[0]
                    #coded_sp_norm = np.array([coded_sp_norm])
                    #coded_sp_norm = np.expand_dims(coded_sp_norm, axis=-1)
                coded_sp_converted_norm = sess.run(x_t, feed_dict={x_test:np.array([coded_sp_norm]), y_A:A_id, y_B:B_id})[0]
                #print(coded_sp_converted_norm[np.isnan(coded_sp_converted_norm)])
                #coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm,axis=-1)
                    #coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm,axis=0)
                if coded_sp_converted_norm.shape[1] > len(f0):
                        coded_sp_converted_norm = coded_sp_converted_norm[:, :-1]
                coded_sp_converted = coded_sp_converted_norm * coded_sps_B_std + coded_sps_B_mean

                coded_sp_converted = coded_sp_converted.T
                coded_sp_converted = np.ascontiguousarray(coded_sp_converted)

                print(np.shape(coded_sp_converted))

                decoded_sp_converted = world_decode_spectral_envelop(coded_sp=coded_sp_converted, fs=sampling_rate)
                #print(decoded_sp_converted[np.isnan(decoded_sp_converted)])
                wav_transformed = world_speech_synthesis(f0=f0_converted, decoded_sp=decoded_sp_converted, ap=ap, fs=sampling_rate,
                                                        frame_period=frame_period)
                #wav_transformed = wav_transformed[:-pad_length]
                print(len(wav_transformed[np.isnan(wav_transformed)]))
                #wav_transformed[np.isnan(wav_transformed)] = 0
                wav_transformed = np.nan_to_num(wav_transformed)
                #print(wav_transformed)
                #librosa.output.write_wav(os.path.join(validation_A_output_dir, os.path.basename(file)), wav_transformed,
                #                        sampling_rate)
                soundfile.write(os.path.join(validation_A_output_dir, os.path.basename(file)), wav_transformed, sampling_rate)

if __name__ == '__main__':
      parser = argparse.ArgumentParser(description='Convert voices using pre-trained CycleGAN model.')

      model_dir = './logdir/train/all_model_v2/model.ckpt-25850'
      source_speaker = 'SF1'
      target_speaker = 'TM3'
      source_language = 'eng'
      target_language = 'kor'

      parser.add_argument('--model_dir', type=str, help='Directory for the pre-trained model.', default=model_dir)
      parser.add_argument('--source_speaker', type=str, help='source_speaker', default=source_speaker)
      parser.add_argument('--target_speaker', type=str, help='target_speaker', default=target_speaker)
      parser.add_argument('--source_language', type=str, help='source_language', default=source_language)
      parser.add_argument('--target_language', type=str, help='target_language', default=target_language)

      argv = parser.parse_args()

      model_dir = argv.model_dir
      source_speaker = argv.source_speaker
      target_speaker = argv.target_speaker
      source_language = argv.source_language
      target_language = argv.target_language

      # conversion(model_dir = model_dir, source=source_speaker, target=target_speaker)
      conversion(model_dir = model_dir, source=source_language, target=target_language)
