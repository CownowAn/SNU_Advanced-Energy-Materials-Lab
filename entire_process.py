from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models from tensorflow.keras import backend as K from sklearn.model_selection import KFold import os
from tqdm.notebook import tqdm

gpunum =1
gpus = tf.config.list_physical_devices('GPU') 
if gpus:
  try:
    tf.config.set_visible_devices(gpus[gpunum], 'GPU') 
    tf.config.experimental.set_virtual_device_configuration(gpus[gpunum],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit =5000)]) logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e: 
    print(e)
    
radius =1
n_bits =2048 
l2_value =0.003 
dropout_rate =0.1 
lr =0.003 
batch_size =30000 
epochs =5000
df = pd.read_excel('/home/thgus4425/data/EA_fromMPdata.xlsx', index_col =0) 
smiles = df['smiles']
labels = df['EA']
fplist = []
done_index = []

for i,s in enumerate(tqdm(smiles)): 
  try:
    mol = Chem.MolFromSmiles(s)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits) fplist.append(fp)
    done_index.append(i)
  except: 
    pass
  
labels = np.array(labels[done_index])
fps = np.array(fplist)
maxval, minval = (9.867671107321804, -3.4570409156131063) np.max(labels), np.min(labels)

def build_model():
  l2 = keras.regularizers.l2(l=0.003)
  randomnormal = keras.initializers.RandomNormal inputs = layers.Input((2048,))
  out = layers.Dense(1024, activation ='relu', kernel_regularizer = l2, kernel_initializer = randomnormal)(inputs)
  out = layers.Dropout(0.2)(out) 
  out = layers.Dense(512, activation ='relu', kernel_regularizer = l2, kernel_initializer = randomnormal)(out)
  out = layers.Dropout(0.2)(out) 
  out = layers.Dense(256, activation ='relu', kernel_regularizer = l2, kernel_initializer = randomnormal)(out)
  out = layers.Dropout(0.2)(out) 
  out = layers.Dense(1, kernel_regularizer = l2, kernel_initializer = randomnormal)(out) 
  
  model = models.Model(inputs,out)
  
  return model

kf = KFold(n_splits =8, random_state =123, shuffle =True) 
kf_results = kf.split(fps)

def TrueMAE(y_true, y_pred):
  truemae = tf.math.reduce_mean(tf.math.abs((tf.squeeze(y_true)-tf.squeeze(y_pred))*(maxval-min val)))
  return truemae

for i, (train_idx, validation_idx) in enumerate(kf.split(fps)):
  K.clear_session() 
  model = build_model() 
  if i ==0:
    model.save_weights('./initial_weights.hdf5') 
  else:
    model.load_weights('./initial_weights.hdf5')
  model.compile(keras.optimizers.Adam(lr), loss = keras.losses.MeanSquaredError(), metrics = [TrueMAE])

def train_generator():
  for fp, l in zip(fps[train_idx], labels[train_idx]):
    yield fp, l 
    
def val_generator():
  for fp, l in zip(fps[validation_idx], labels[validation_idx]): 
    yield fp, l

def minmax_norm(inputs, labels):
  labels = (labels-minval)/(maxval-minval) 
  return inputs, labels

types = (tf.int32, tf.float32)
shapes = ((2048,),())
train_data = tf.data.Dataset.from_generator(train_generator, output_types = types, output_shapes = shapes) 
train_data = train_data.map(minmax_norm)
train_data = train_data.batch(batch_size)
val_data = tf.data.Dataset.from_generator(val_generator, output_types = types, output_shapes = shapes)
val_data = val_data.map(minmax_norm) 
val_data = val_data.batch(batch_size)

tensorboard = keras.callbacks.TensorBoard('./logs{}'.format(i)) 
if not os.path.isdir('./checkpoint{}'.format(i)):
  os.mkdir('./checkpoint{}'.format(i)) 
checkpointcallback = tf.keras.callbacks.ModelCheckpoint('./checkpoint{}'.format(i)+'/epoch{epoch:05d}_val_ TrueMAE{val_TrueMAE:.4f}.hdf5', monitor ='val_TrueMAE', save_best_only =True, save_weights_only =True, mode ='min')

def schedule(epoch, _):
  if epoch<500:
    return lr
  elif epoch>=500:
    return lr * tf.math.exp(3* (500- epoch)/(epochs-500)) 
  learningratecallback = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0) 
  model.fit(train_data, validation_data = val_data, epochs = epochs, callbacks = [tensorboard, checkpointcallback, learningratecallback])
