import json
import os
from importlib import import_module
import numpy as np
import tensorflow as tf
from utility import *
from util.wrapper import validate_log_dirs


args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'corpus_name', 'vcc2016', 'Corpus name')
tf.app.flags.DEFINE_string(
    'logdir_root', None, 'root of log dir')
tf.app.flags.DEFINE_string(
    'logdir', None, 'log dir')
tf.app.flags.DEFINE_string(
    'restore_from', None, 'restore from dir (not from *.ckpt)')
tf.app.flags.DEFINE_string('gpu_cfg', None, 'GPU configuration')
tf.app.flags.DEFINE_integer('summary_freq', 1000, 'Update summary')
tf.app.flags.DEFINE_string(
    'ckpt', None, 'specify the ckpt in restore_from (if there are multiple ckpts)')  # TODO
tf.app.flags.DEFINE_string(
    'architecture', 'architecture-vawgan-vcc2016.json', 'network architecture')

tf.app.flags.DEFINE_string('model_module', 'model.vawgan_multi_decoder', 'Model module')
tf.app.flags.DEFINE_string('model', 'VAWGAN', 'Model: ConvVAE, VAWGAN')

tf.app.flags.DEFINE_string('trainer_module', 'trainer.vawgan_train_multi_decoder', 'Trainer module')
tf.app.flags.DEFINE_string('trainer', 'VAWGANTrainer', 'Trainer: VAETrainer, VAWGANTrainer')



def main(unused_args=None):
   
    module = import_module(args.trainer_module, package=None)
    TRAINER = getattr(module, args.trainer)


    dirs = validate_log_dirs(args)
    tf.gfile.MakeDirs('./logdir/train/all_model_v2/')

    with open(args.architecture) as f:
        arch = json.load(f)

    with open(os.path.join('./logdir/train/all_model_v2/', args.architecture), 'w') as f:
        json.dump(arch, f, indent=4)

    BATCHSIZE = 8
    
    trainer = TRAINER(arch, args, './logdir/train/all_model_v2/',num_features=36,frames=128)
    trainer.train(nIter=arch['training']['max_iter'],batchsize=BATCHSIZE)


if __name__ == '__main__':
    
    main()
