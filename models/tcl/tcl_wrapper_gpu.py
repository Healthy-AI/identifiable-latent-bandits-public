### Wrapper function for TCL 
#
# this code is adapted from: https://github.com/hirosm/TCL
#
#
import os

import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.decomposition import FastICA

from .tcl_core import inference
from .tcl_core import train_gpu as train
from .tcl_eval import get_tensor, calc_accuracy
from models.preprocessing import pca
from utils import load_pickle, save_pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def TCL_wrapper(sensor, label, list_hidden_nodes, random_seed=0, max_steps=int(7e4), max_steps_init=int(7e4),
                ckpt_dir='./', test=False, latent_comp=None, is_csv=False, **kwargs):
    # Training ----------------------------------------------------
    initial_learning_rate = 0.01  # initial learning rate
    momentum = 0.9  # momentum parameter of SGD
    # max_steps = int(7e4) # number of iterations (mini-batches)
    decay_steps = int(5e4)  # decay steps (tf.train.exponential_decay)
    decay_factor = 0.1  # decay factor (tf.train.exponential_decay)
    batch_size = 512  # mini-batch size
    moving_average_decay = 0.9999  # moving average decay of variables to be saved
    checkpoint_steps = 1e5  # interval to save checkpoint
    num_comp = sensor.shape[0] if latent_comp is None else latent_comp  # number of latent components

    # for MLR initialization
    decay_steps_init = int(5e4)  # decay steps for initializing only MLR

    # Other -------------------------------------------------------
    train_dir = ckpt_dir  # save directory

    num_segment = len(np.unique(label))

    # Preprocessing -----------------------------------------------
    if not is_csv:
        if test:
            pca_parm = load_pickle(os.path.join(ckpt_dir, 'pca_params.bin'))
            sensor, _ = pca(sensor, num_comp=num_comp, params=pca_parm)
        else:
            sensor, pca_parm = pca(sensor, num_comp=num_comp)

    if not test:
        # Train model (only MLR) --------------------------------------
        train(sensor,
              label,
              num_class=len(np.unique(label)),  # num_segment,
              list_hidden_nodes=list_hidden_nodes,
              initial_learning_rate=initial_learning_rate,
              momentum=momentum,
              max_steps=max_steps_init,  # For init
              decay_steps=decay_steps_init,  # For init
              decay_factor=decay_factor,
              batch_size=batch_size,
              train_dir=train_dir,
              checkpoint_steps=checkpoint_steps,
              moving_average_decay=moving_average_decay,
              MLP_trainable=False,  # For init
              save_file='model_init.ckpt',  # For init
              random_seed=random_seed,
              n=kwargs.get('n', 0),
              l=kwargs.get('l', 0))

        init_model_path = os.path.join(train_dir, 'model_init.ckpt')

        # Train model -------------------------------------------------
        train(sensor,
              label,
              num_class=len(np.unique(label)),  # num_segment,
              list_hidden_nodes=list_hidden_nodes,
              initial_learning_rate=initial_learning_rate,
              momentum=momentum,
              max_steps=max_steps,
              decay_steps=decay_steps,
              decay_factor=decay_factor,
              batch_size=batch_size,
              train_dir=train_dir,
              checkpoint_steps=checkpoint_steps,
              moving_average_decay=moving_average_decay,
              load_file=init_model_path,
              random_seed=random_seed,
              n=kwargs.get('n', 0),
              l=kwargs.get('l', 0))

    # now that we have trained everything, we can evaluate results:
    eval_dir = ckpt_dir
    ckpt = tf.train.get_checkpoint_state(eval_dir)

    with tf.Graph().as_default():
        data_holder = tf.placeholder(tf.float32, shape=[None, sensor.shape[0]], name='data')

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits, feats = inference(data_holder, list_hidden_nodes, num_class=num_segment)

        # Calculate predictions.
        top_value, preds = tf.nn.top_k(logits, k=1, name='preds')

        # Restore the moving averaged version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            saver.restore(sess, ckpt.model_checkpoint_path)

            tensor_val = get_tensor(sensor, [preds, feats], sess, data_holder, batch=256)
            pred_val = tensor_val[0].reshape(-1)
            feat_val = tensor_val[1]

    # Calculate accuracy ------------------------------------------
    accuracy, confmat = calc_accuracy(pred_val, label)

    # Apply fastICA -----------------------------------------------
    if test:
        ica = load_pickle(os.path.join(ckpt_dir, 'fast_ica.bin'))
        feat_val_ica = ica.transform(feat_val)
    else:
        ica = FastICA(random_state=random_seed).fit(feat_val)
        feat_val_ica = ica.transform(feat_val)
        save_pickle(os.path.join(ckpt_dir, 'fast_ica.bin'), ica)
        if not is_csv:
            save_pickle(os.path.join(ckpt_dir, 'pca_params.bin'), pca_parm)

    feat_val_ica = feat_val_ica.T  # Estimated feature
    feat_val = feat_val.T

    return feat_val, feat_val_ica, accuracy


def inference_tcl(sensor, label, list_hidden_nodes, ckpt_dir='./', is_csv=False):
    """Infernce function used by the Bandits."""
    # Training ----------------------------------------------------
    moving_average_decay = 0.9999  # moving average decay of variables to be saved
    num_comp = sensor.shape[0]
    num_segment = len(np.unique(label))
    assert sensor.ndim == 2, 'x has to have a dim of 2' 

    # Preprocessing -----------------------------------------------
    if not is_csv:
        pca_params = load_pickle(os.path.join(ckpt_dir, 'pca_params.bin'))
        sensor, _ = pca(sensor, num_comp=num_comp, params=pca_params)

    # now that we have trained everything, we can evaluate results:
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)

    with tf.Graph().as_default():
        data_holder = tf.placeholder(tf.float32, shape=[None, sensor.shape[0]], name='data')

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits, feats = inference(data_holder, list_hidden_nodes, num_class=num_segment)

        # Calculate predictions.
        top_value, preds = tf.nn.top_k(logits, k=1, name='preds')

        # Restore the moving averaged version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            saver.restore(sess, ckpt.model_checkpoint_path)
            tensor_val = get_tensor(sensor, [preds, feats], sess, data_holder, batch=256)
            pred_val = tensor_val[0].reshape(-1)
            feat_val = tensor_val[1]
    return feat_val


class TCLInference:

    def __init__(self, list_hidden_nodes, ckpt_dir='./'):
        """Initialize and load the model only once."""
        self.moving_average_decay = 0.9999 # constant and fixed
        self.ckpt_dir = ckpt_dir
        self.list_hidden_nodes = list_hidden_nodes
        self._is_built = False

    def build_graph(self):
        """Infernce function used by the Bandits."""
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.data_holder = tf.placeholder(tf.float32, shape=[None, self.num_comp], name='data')
            logits, self.feats = inference(self.data_holder, self.list_hidden_nodes, num_class=self.num_segment)
            top_value, self.preds = tf.nn.top_k(logits, k=1, name='preds')
            variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            self.sess = tf.Session(graph=self.graph)
            saver.restore(self.sess, ckpt.model_checkpoint_path)

    def __call__(self, sensor):
        """Runs inference on the given sensor data without reloading the model.

        Pre and post processing steps like PCA and ICA are not done here see bandits.inference_models.LVMInstance
        """
        assert sensor.ndim == 2, 'x has to have a dim of 2' 
        if not self._is_built:
            self.num_comp = sensor.shape[0]
            self.num_segment = 1 # Place holder: only used in MLR, has no effect on the latent state.
            self.build_graph()
            self._is_built = True
        with self.graph.as_default():
            tensor_val = get_tensor(sensor, [self.preds, self.feats], self.sess, self.data_holder, batch=256)
            pred_val = tensor_val[0].reshape(-1) # MLR result
            feat_val = tensor_val[1]
        return feat_val

    def close(self):
        """Closes the TensorFlow session to free resources."""
        self.sess.close()
