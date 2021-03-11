# import the required libraries
import numpy as np
import time
import random
import pickle
import codecs
import collections
import os
import math
import json
import tensorflow as tf
from six.moves import xrange
import six

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from shapely.geometry import Polygon


# libraries required for visualisation:
from IPython.display import SVG, display
import PIL
from PIL import Image
import matplotlib.pyplot as plt

# set numpy output to something sensible
np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

import svgwrite # conda install -c omnia svgwrite=1.1.6


tf.logging.info("TensorFlow Version: %s", tf.__version__)

# import our command line tools
# from sketch_rnn_train import *
from model_two import *
import model_two as sketch_rnn_model
import utils 
from rnn import *


# define the path of the model you want to load, and also the path of the dataset

data_dir = './'
data_name = 'Data_new5.npz'
models_root_dir = './'
model_dir = 'train_without_all_scale'

def load_dataset(data_dir,data_name, model_params, inference_mode=False):
  """Loads the .npz file, and splits the set into train/valid/test."""
  data_filepath = os.path.join(data_dir, data_name)
  if six.PY3:
    data = np.load(data_filepath, encoding='latin1', allow_pickle=True)
  else:
    data = np.load(data_filepath, allow_pickle=True)
  train_strokes = data['train']
  valid_strokes = data['valid']
  test_strokes = data['test']
 

  eval_model_params = sketch_rnn_model.copy_hparams(model_params)

  eval_model_params.use_input_dropout = 0
  eval_model_params.use_recurrent_dropout = 0
  eval_model_params.use_output_dropout = 0
  eval_model_params.is_training = 1

  if inference_mode:
    eval_model_params.batch_size = 1
    eval_model_params.is_training = 0

  sample_model_params = sketch_rnn_model.copy_hparams(eval_model_params)
  sample_model_params.batch_size = 1  # only sample one at a time
  sample_model_params.max_seq_len = 1  # sample one point at a time

  train_set = utils.DataLoader(
      train_strokes,
      model_params.batch_size,
      max_seq_length=model_params.max_seq_len,
      random_scale_factor=0.0,
      augment_stroke_prob=model_params.augment_stroke_prob)

  normalizing_scale_factor = train_set.calculate_normalizing_scale_factor()
  train_set.normalize(normalizing_scale_factor)

  valid_set = utils.DataLoader(
      valid_strokes,
      eval_model_params.batch_size,
      max_seq_length=eval_model_params.max_seq_len,
      random_scale_factor=0.0,
      augment_stroke_prob=0.0)
  valid_set.normalize(normalizing_scale_factor)

  test_set = utils.DataLoader(
      test_strokes,
      eval_model_params.batch_size,
      max_seq_length=eval_model_params.max_seq_len,
      random_scale_factor=0.0,
      augment_stroke_prob=0.0)
  test_set.normalize(normalizing_scale_factor)
  tf.logging.info('normalizing_scale_factor %4.4f.', normalizing_scale_factor)
  result = [
      train_set, valid_set, test_set, model_params, eval_model_params,
      sample_model_params
  ]
  return result




def load_env_compatible(data_dir, data_name, model_dir):
  """Loads environment for inference mode, used in jupyter notebook."""
  # modified https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/sketch_rnn_train.py
  # to work with depreciated tf.HParams functionality
  model_params = sketch_rnn_model.get_default_hparams()
  with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
    data = json.load(f)
  fix_list = ['conditional', 'is_training', 'use_input_dropout', 'use_output_dropout', 'use_recurrent_dropout']
  for fix in fix_list:
    data[fix] = (data[fix] == 1)
  model_params.parse_json(json.dumps(data))
  return load_dataset(data_dir,data_name, model_params, inference_mode=True)


[train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model] = load_env_compatible(data_dir, data_name, model_dir)



def reset_graph():
  """Closes the current default session and resets the graph."""
  sess = tf.get_default_session()
  if sess:
    sess.close()
  tf.reset_default_graph()


# construct the sketch-rnn model here:
reset_graph()
model = Model(hps_model)
eval_model = Model(eval_hps_model, reuse=True)
sample_model = Model(sample_hps_model, reuse=True)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())



# loads the weights from checkpoint into our model
# load_checkpoint(sess, model_dir)
saver = tf.train.Saver(tf.global_variables())
saver.restore(sess, "./train_without_all_scale/vector-23000")



model_dir



eval_model.hps.max_seq_len


# We define two convenience functions to encode a stroke into a latent vector, and decode from latent vector to stroke.


def encode(input_strokes):
  strokes = utils.to_big_strokes(input_strokes, max_len=eval_model.hps.max_seq_len).tolist()
  strokes.insert(0, [0, 0,0,0,0, 1, 0, 0])
  seq_len = [len(input_strokes)]
  return sess.run(eval_model.batch_z, feed_dict={eval_model.input_data: [strokes], eval_model.sequence_lengths: seq_len})[0]

def decode(z_input=None, draw_mode=True, temperature=0.1, factor=0.05):
  z = None
  if z_input is not None:
    z = [z_input]
  sample_strokes = sample(sess, sample_model, seq_len=eval_model.hps.max_seq_len, temperature=temperature, z=z)
  strokes = utils.to_normal_strokes(sample_strokes)
  return sample_strokes, strokes


def two2four(X):
    x0=X[0]
    y0=X[1]
    x1=X[2]
    y1=X[3]
    h=X[4]
    K=1+(y0-y1)**2/(x0-x1)**2
    y2=y1+h/np.sqrt(K)
    x2=x1+h*np.sqrt(1-1/K)
    
    x3=x0-x1+x2
    y3=y0-y1+y2
    

    poly=np.array([x0,x1,x2,x3,y0,y1,y2,y3])   
    return poly



def convert_ploy(x):
    polyset=[]
    for i in range(x.shape[0]):
        poly=two2four(x[i])
        polyset.append(poly)
    return polyset

def draw_poly(ax, pc):
    for i in range(len(pc)):
        X, Y= pc[i][0:4],pc[i][4:8]
#         ax.scatter(X, Y)
        ax.plot(X, Y,c='b')
        ax.plot([X[-1],X[0]], [Y[-1],Y[0]], c='b')
        ax.axis('equal')
        my_x_ticks = np.arange(-0.2, 1.2, 0.2)
        my_y_ticks = np.arange(-0.2, 1.2, 0.2)
        plt.xticks(my_x_ticks)
        plt.yticks(my_y_ticks)


def plot_polygon_noend(y,m=1,n=1, save=False, savename='pclouds'):
    fig = plt.figure(figsize=(15*m,15*n))
    fig.set_tight_layout(True)
   
    for i in range(n):
        for j in range(m):
            
            idx = i*m + j
            sample=convert_ploy(y)
            ax = fig.add_subplot(n, m, idx+1, xticks=[], yticks=[])
            draw_poly(ax, sample)
    if save:
        plt.savefig(savename, bbox_inches='tight', pad_inches=0)
        plt.close('all')
        plt.close(fig)


# In[23]:


def trans_ori(o):
    o2 = np.zeros(o.shape)
    o2[0] = o[0]
    for i in range(1, len(o)):
        o2[i,:5] = o2[i-1,:5]+o[i,:5]
        o2[i,5:] = o[i,5:]
#         o2[i] = o[i-1]+o[i]
    return o2


# get a sample drawing from the test set, and render it to .svg
stroke = test_set.random_sample()
# stroke = train_set.random_sample()
Stroke=trans_ori(stroke)


plot_polygon_noend(Stroke)
# Let's try to encode the sample stroke into latent vector $z$
z = encode(stroke)
so, o = decode(z, temperature=0.8) # convert z back to drawing at temperature of 0.8
ooooo=trans_ori(so)
plot_polygon_noend(ooooo)

stroke_list = []
for i in range(10):
    for _ in range(10):
      stroke_list.append(decode(z, draw_mode=False, temperature=0.1*i+0.1)[0])

def subplot_num(m, i, j):
    return i*m + j

def plot_polygonset(samples, n, m, save=False, savename='pclouds'):
    fig = plt.figure(figsize=(10*m,10*n))
    fig.set_tight_layout(True)
    for i in range(n):
        for j in range(m):
            idx = subplot_num(m, i, j) 
#             print(i,j,m,n,idx)
            ax = fig.add_subplot(n, m, idx+1, xticks=[], yticks=[])
            
            o = samples[idx]
            index = np.arange(len(o))[o[:,2] == 1]
            if(len(index)>1):
                poly = trans_ori(o)
                poly=convert_ploy(poly)
                draw_poly(ax, poly[:index[0]+1])
                for k in range(len(index)-1):
                    draw_poly(ax, poly[index[k]+1:index[k+1]+1])
            else:
                poly = trans_ori(o)
                poly=convert_ploy(poly)
                draw_poly(ax, poly)
    if save:
        plt.savefig(savename, bbox_inches='tight', pad_inches=0)
    plt.show()


plot_polygonset(stroke_list, 10, 10)




