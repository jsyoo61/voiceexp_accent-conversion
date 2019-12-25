import logging
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from util.wrapper import save
from model.vawgan_multi_decoder import encode, decode,discriminate
from util.layers import GaussianLogDensity, GaussianKLD, \
    GaussianSampleLayer, lrelu
from utility import *
from util.wrapper import load
from speech_tools import load_pickle, sample_train_data

def lr_schedule(step, schedule):
    for s, lr in schedule:
        if step < s:
            return 1e-5
    return 1e-5

class VAWGANTrainer(object):
    def __init__(self,arch, args, dirs, num_features, frames):

        self.arch = arch
        self.args = args
        self.dirs = dirs
        self.input_shape = [None, num_features, None]
        self.x_A = tf.placeholder(tf.float32, self.input_shape, name='input_real_source')
        self.x_B = tf.placeholder(tf.float32, self.input_shape, name='input_real_target')

        self.y_A = tf.placeholder(tf.float32, shape = [2], name = 'A_lang_vector')
        self.y_B = tf.placeholder(tf.float32, shape = [2], name = 'B_lang_vector')

        # self.y_A = tf.placeholder(tf.float32, shape = [10], name = 'A_id_vector')
        # self.y_B = tf.placeholder(tf.float32, shape = [10], name = 'B_id_vector')

        self.encoder = encode
        self.decoder = decode
        self.discriminator = discriminate
        self.build_model()
        self.opt = self._optimize()
        logging.basicConfig(
            level=logging.INFO,
            filename=os.path.join(dirs, 'training.log')
        )
    def build_model(self):
        print("build model!")
        is_training =True

        def circuit_loop(x, y):

            # Reconstructing x
            # x,y ==[encoder]==> z_mu, z_log_var
            # z_mu, z_log_var ==[Gaussian Sample]==> z
            # z,y ==[decoder]==> xh
            # x,y ==[discriminator]==> x_logit

            z_mu, z_lv = self.encoder(x, y, is_training = is_training, scope_name = 'encoder')
            z = GaussianSampleLayer(z_mu, z_lv)
            xh = self.decoder(z, y, is_training = is_training, scope_name = 'decoder')
            x_logit = self.discriminator(x, y, is_training = is_training, scope_name = 'discriminator')
            result = dict()
            result['z'] =z
            result['z_mu']=z_mu
            result['z_lv']=z_lv
            result['xh']=xh
            result['x_logit']=x_logit
            return result

        def cycle_loop(src_x, src_y,trg_x,trg_y):

            # Reconstructing x with cycle loop
            # x_src, y_src ==[encoder]==> z_mu, z_log_var
            # z_mu, z_log_var ==[Gaussian Sample]==> z
            # z, y_trg ==[decoder]==> xh

            z_mu, z_lv = self.encoder(src_x, src_y, is_training = is_training, scope_name = 'encoder')
            z = GaussianSampleLayer(z_mu, z_lv)
            xh = self.decoder(z, trg_y, is_training = is_training, scope_name = 'decoder')
            z_mu_cycle, z_lv_cycle = self.encoder(xh, trg_y, is_training = is_training, scope_name = 'encoder')
            z_cycle = GaussianSampleLayer(z_mu_cycle, z_lv_cycle)
            xh_cycle = self.decoder(z_cycle, src_y, is_training = is_training, scope_name = 'decoder')
            xh_logit = self.discriminator(xh, trg_y, is_training = is_training, scope_name = 'discriminator')

            result = dict()
            result['z'] =z
            result['z_mu']=z_mu
            result['z_mu_cycle']=z_mu_cycle
            result['z_lv']=z_lv
            result['z_lv_cycle']=z_lv_cycle
            result['xh']=xh
            result['xh_cycle']=xh_cycle
            result['xh_logit']=xh_logit
            return result

        cycle_s = cycle_loop(self.x_A, self.y_A,self.x_B, self.y_B)
        cycle_t = cycle_loop(self.x_B, self.y_B,self.x_A, self.y_A)

        recon_s = circuit_loop(self.x_A, self.y_A)
        recon_t = circuit_loop(self.x_B, self.y_B)


        hyperp = self.arch['training']
        k = tf.constant(hyperp['clamping'], shape=[])

        self.loss = dict()

        self.loss['conv_s2t'] = \
                  tf.reduce_mean(recon_t['x_logit']) \
                - tf.reduce_mean(cycle_s['xh_logit'])
        self.loss['conv_s2t'] /= k

        self.loss['WGAN'] = self.loss['conv_s2t']

        self.loss['KL(z)'] = \
                tf.reduce_mean(
                    GaussianKLD(
                        recon_s['z_mu'], recon_s['z_lv'],
                        tf.zeros_like(recon_s['z_mu']), tf.zeros_like(recon_s['z_lv']))) +\
                tf.reduce_mean(
                    GaussianKLD(
                        recon_t['z_mu'], recon_t['z_lv'],
                        tf.zeros_like(recon_t['z_mu']), tf.zeros_like(recon_t['z_lv']))) +\
                tf.reduce_mean(
                    GaussianKLD(
                        cycle_s['z_mu_cycle'], cycle_s['z_lv_cycle'],
                        tf.zeros_like(cycle_s['z_mu_cycle']), tf.zeros_like(cycle_s['z_lv_cycle']))) +\
                tf.reduce_mean(
                    GaussianKLD(
                        cycle_t['z_mu_cycle'], cycle_t['z_lv_cycle'],
                        tf.zeros_like(cycle_t['z_mu_cycle']), tf.zeros_like(cycle_t['z_lv_cycle'])))

        self.loss['KL(z)'] /= 4.0

        self.loss['Dis'] = \
                tf.reduce_mean(
                    GaussianLogDensity(
                        slim.flatten(self.x_A),
                        slim.flatten(cycle_s['xh_cycle']),
                        tf.zeros_like(slim.flatten(self.x_A)))) +\
                tf.reduce_mean(
                    GaussianLogDensity(
                        slim.flatten(self.x_B),
                        slim.flatten(cycle_t['xh_cycle']),
                        tf.zeros_like(slim.flatten(self.x_B)))) +\
                tf.reduce_mean(
                    GaussianLogDensity(
                        slim.flatten(self.x_A),
                        slim.flatten(recon_s['xh']),
                        tf.zeros_like(slim.flatten(self.x_A)))) +\
                tf.reduce_mean(
                    GaussianLogDensity(
                        slim.flatten(self.x_B),
                        slim.flatten(recon_t['xh']),
                        tf.zeros_like(slim.flatten(self.x_B))))

        self.loss['Dis'] /= - 4.0

        trainables = tf.trainable_variables()

        self.r = tf.placeholder(shape=[], dtype=tf.float32)

        self.e_vars = [v for v in trainables if 'encoder' in v.name]
        self.g_vars = [v for v in trainables if 'decoder' in v.name]
        self.d_vars = [v for v in trainables if 'discriminator' in v.name]

        self.obj_Dx = - self.loss['conv_s2t'] * k
        self.obj_Gx = self.r * self.loss['conv_s2t'] + self.loss['Dis']
        self.obj_Ez = self.loss['KL(z)'] + self.loss['Dis']


    def _optimize(self):
        hyperp = self.arch['training']
        global_step = tf.Variable(0, name = 'global_step')
        lr = tf.placeholder(dtype = tf.float32, name = 'learning_rate')
        k = tf.constant(hyperp['clamping'], shape = [])
        optimizer_d = tf.train.AdamOptimizer(lr,0.5)
        optimizer_g = tf.train.AdamOptimizer(lr,0.5)

        r = tf.placeholder(shape=[], dtype=tf.float32)

        opt_d = optimizer_d.minimize(self.obj_Dx, var_list=self.d_vars)
        opt_ds = [opt_d]
        logging.info('The following variables are clamped:')
        with tf.control_dependencies(opt_ds):
            with tf.name_scope('Clamping'):
                for v in self.d_vars:
                    v_clamped = tf.clip_by_value(v, -k, k)
                    clamping = tf.assign(v, v_clamped)
                    opt_ds.append(clamping)
                    logging.info(v.name)


        opt_g = optimizer_g.minimize(self.obj_Gx, var_list=self.g_vars, global_step=global_step)
        opt_e = optimizer_g.minimize(self.obj_Ez, var_list=self.e_vars)

        return dict(
            d=opt_ds,
            g=opt_g,
            gz=opt_e,
            lr=lr,
            gamma=r,
            global_step=global_step)

    def train(self, nIter, machine=None, summary_op=None,batchsize=16):
        num_mcep = 36

        frame_period = 5.0
        n_frames = 128
        # num_iter_per_epoch = 20

        vae_saver = tf.train.Saver(max_to_keep=None)
        sv = tf.train.Supervisor(
            logdir=self.dirs,
            save_model_secs=600,
            global_step=self.opt['global_step'])

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True))

        exp_dir = os.path.join('processed')
        hyperp = self.arch['training']
        info = {'nEpoch': hyperp['epoch_vae']}

        def print_log(info, results):

            msg = 'Epoch [{:3d}/{:3d}] '.format(info['ep'], info['nEpoch'])
            msg += '[{:4d}/{:4d}] '.format(info['it'], info['nIter'])
            msg += 'W: {:.2f} '.format(results['conv_s2t'])
            msg += 'DIS={:5.2f}, KLD={:5.2f}'.format(results['Dis'], results['KL(z)'])
            print('\r{}'.format(msg), flush=True)
            logging.info(msg)


        print('Loading cached data...')

        with sv.managed_session(config=sess_config) as sess:
            load(vae_saver, sess, "./logdir/train/all_model_v2/", ckpt=None)
            try:
                update_G_E = [self.opt['g'], self.opt['gz']]
                # speaker_list = ['SF1','SF2','SF3','SM1','SM2','TF1','TF2','TM1','TM2','TM3']
                language_list = ['eng', 'kor']
                num_language = len(language_list)

                #VAE training
                for ep in range( hyperp['epoch_vae'] ):

                    lr = lr_schedule(ep, hyperp['lr_schedule'])
                    info.update({'ep': ep + 1})

                    # Train every possible combinations. n^2
                    # i : Source
                    for i in range( num_language ):

                        # Create one-hot vector
                        A_id = [0.] * num_language
                        A_id[i] = 1.0

                        # Load parameters for source
                        src_language = language_list[i]
                        # src_speaker = speaker_list[i]

                        exp_A_dir = os.path.join(exp_dir, src_language)
                        # exp_A_dir = os.path.join(exp_dir, src_speaker)

                        coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A = load_pickle(
                                    os.path.join(exp_A_dir, 'cache{}.p'.format(num_mcep)))

                        # j : Target
                        for j in range( num_language ):

                            # Create one-hot vector
                            B_id = [0.] * num_language
                            B_id[j] = 1.0

                            # Load parameters for target
                            trg_language = language_list[j]
                            # trg_speaker = speaker_list[j]

                            exp_B_dir = os.path.join(exp_dir, trg_language)
                            # exp_B_dir = os.path.join(exp_dir, trg_speaker)

                            coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std, log_f0s_mean_B, log_f0s_std_B = load_pickle(
                                        os.path.join(exp_B_dir, 'cache{}.p'.format(num_mcep)))

                            # Sample training data
                            dataset_A, dataset_B = sample_train_data(dataset_A=coded_sps_A_norm, dataset_B=coded_sps_B_norm, n_frames=n_frames)
                            num_dataset = dataset_A.shape[0]
                            num_iter_per_epoch = num_dataset // batchsize
                            info.update({'nIter': num_iter_per_epoch})

                            print('dataset_A:%s, dataset_B:%s'%(str(dataset_A.shape), str(dataset_B.shape)))
                            print('Training %s to %s'%(src_language, trg_language))

                            # Things to fetch in session
                            fetches = {
                                'conv_s2t': self.loss['conv_s2t'],
                                'Dis': self.loss['Dis'],
                                'KL(z)': self.loss['KL(z)'],
                                'opt_g': self.opt['g'],
                                'opt_e': self.opt['gz'],
                                'step': self.opt['global_step'],
                            }


                            # Training
                            for it in range(num_iter_per_epoch):

                                # Set batch range
                                start = it * batchsize
                                end = (it + 1) * batchsize

                                # Training Session
                                feed_dict = {self.r : 0., self.opt['lr'] : lr, self.x_A : dataset_A[start:end], self.x_B : dataset_B[start:end], self.y_A : A_id, self.y_B : B_id}
                                results = sess.run(fetches, feed_dict=feed_dict)

                                # Print log every 5 iterations
                                if (it + 1) % 5 == 0 :
                                    info.update({'it': it + 1})
                                    print_log(info, results)

                    # Save model every 10 epochs
                    if (ep + 1) % 10 == 0:
                        print('saving model...')
                        save(vae_saver, sess, os.path.join(self.dirs, 'VAE'), fetches['step'])

                info.update({'nEpoch': hyperp['epoch_vawgan'] + hyperp['epoch_vae']})

                #VAE - WGAN training
                for ep in range(150):

                    lr = lr_schedule(ep, hyperp['lr_schedule'])
                    info.update({ 'ep': ep + hyperp['epoch_vae'] + 1 })

                    # Train every possible combinations. n^2
                    # i : Source
                    for i in range( num_language ):

                        # Create one-hot vector
                        A_id = [0.] * num_language
                        A_id[i] = 1.0

                        # Load parameters for source
                        src_language = language_list[i]
                        # src_speaker = speaker_list[i]

                        exp_A_dir = os.path.join(exp_dir, src_language)
                        # exp_A_dir = os.path.join(exp_dir, src_speaker)

                        coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std, log_f0s_mean_A, log_f0s_std_A = load_pickle(
                                    os.path.join(exp_A_dir, 'cache{}.p'.format(num_mcep)))

                        # j : Target
                        for j in range( num_language ):

                            # Create one-hot vector
                            B_id = [0.] * num_language
                            B_id[j] = 1.0

                            # Load parameters for target
                            trg_language = language_list[j]
                            # trg_speaker = speaker_list[j]

                            exp_B_dir = os.path.join(exp_dir, trg_language)
                            # exp_B_dir = os.path.join(exp_dir, trg_speaker)

                            coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std, log_f0s_mean_B, log_f0s_std_B = load_pickle(
                                        os.path.join(exp_B_dir, 'cache{}.p'.format(num_mcep)))

                            # Sample training data
                            dataset_A, dataset_B = sample_train_data(dataset_A=coded_sps_A_norm, dataset_B=coded_sps_B_norm, n_frames=n_frames)
                            num_dataset = dataset_A.shape[0]
                            num_iter_per_epoch = num_dataset // batchsize
                            info.update({'nIter': num_iter_per_epoch})

                            print('Training %s to %s'%(src_language, trg_language))

                            # Training
                            for it in range(num_iter_per_epoch):

                                # Set batch range
                                start = it * batchsize
                                end = (it + 1) * batchsize

                                # Training frequenc
                                # Discrminator : Encoder & Decoder == 4 : 1

                                # Things to fetch in Session:

                                # Train Discriminator (4 times)
                                if (it % 5)!=0:
                                    fetches = {
                                        'conv_s2t': self.loss['conv_s2t'],
                                        'Dis': self.loss['Dis'],
                                        'KL(z)': self.loss['KL(z)'],
                                        'opt_d': self.opt['d'],
                                        'step': self.opt['global_step'],
                                    }

                                # Train Encoder & Decoder (1 times)
                                elif (it % 5)==0:
                                    fetches = {
                                        'conv_s2t': self.loss['conv_s2t'],
                                        'Dis': self.loss['Dis'],
                                        'KL(z)': self.loss['KL(z)'],
                                        'opt_g': self.opt['g'],
                                        'opt_e': self.opt['gz'],
                                        'step': self.opt['global_step'],
                                    }

                                # (Impossible)
                                else:
                                    print('error??')

                                # Training Session
                                feed_dict = {self.r: hyperp['gamma'], self.opt['lr']: lr,self.x_A:dataset_A[start:end],self.x_B:dataset_B[start:end],self.y_A:A_id,self.y_B:B_id}
                                results = sess.run(fetches, feed_dict=feed_dict)

                                # Print log every 5 iterations
                                if (it + 1) % 5 == 0:
                                    info.update({'it': it + 1})
                                    print_log(info, results)

                    # Save model every 10 epochs
                    if (ep + 1) % 10 == 0:
                        print('saving model...')
                        save(vae_saver, sess, os.path.join(self.dirs, 'VAEGAN'), fetches['step'])


            except KeyboardInterrupt:
                print()
            finally:
                save(sv.saver, sess, self.dirs, fetches['step'])
            print()
