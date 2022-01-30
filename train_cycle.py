import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import cv2
import os
import shutil
from time import time
import matplotlib.pyplot as plt
from my_models import generator, latent_generator
from my_utils_cycle import *
from Unet_util import Unet
from convolutional_geneator import pix2pix_generator



tfb = tfp.bijectors
tfd = tfp.distributions

FLAGS, unparsed = flags()

num_epochs = FLAGS.num_epochs
batch_size = FLAGS.batch_size
dataset_g = FLAGS.dataset_g
dataset_f = FLAGS.dataset_f
lr = FLAGS.lr
gpu_num = FLAGS.gpu_num
learntop = bool(FLAGS.learntop)
remove_all = bool(FLAGS.remove_all)
desc = FLAGS.desc
model_depth = FLAGS.model_depth
latent_depth = FLAGS.latent_depth
inv_conv_activation = FLAGS.inv_conv_activation
T = FLAGS.T

ml_threshold = 10


all_experiments = 'experiment_results/'

if os.path.exists(all_experiments) == False:
    os.mkdir(all_experiments)

    
    
exp_path = all_experiments + 'Injective_' + \
    dataset_g + '_' + dataset_f + '_' + 'model_depth_%d' % (model_depth,) + '_' + 'latent_depth_%d'% (latent_depth,) + '_learntop_%d' \
        % (int(learntop)) + '_' + desc


if os.path.exists(exp_path) == True and remove_all == True:
    shutil.rmtree(exp_path)

if os.path.exists(exp_path) == False:
    os.mkdir(exp_path)
    
    

exp_path_g = all_experiments + 'Injective_' + \
    dataset_g + '_' + 'model_depth_%d' % (model_depth,) + '_' + 'latent_depth_%d'% (latent_depth,) + '_learntop_%d' \
        % (int(learntop)) + '_' + dataset_g


exp_path_f = all_experiments + 'Injective_' + \
    dataset_f + '_' + 'model_depth_%d' % (model_depth,) + '_' + 'latent_depth_%d'% (latent_depth,) + '_learntop_%d' \
        % (int(learntop)) + '_' + dataset_f


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[gpu_num], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)



class Prior(layers.Layer):
    """Defines the low dimensional distribution as Guassian"""
    def __init__(self, **kwargs):
        super(Prior, self).__init__()
        
        latent_dim = kwargs.get('latent_dim', 64)
            
        self.mu = tf.Variable(tf.ones(latent_dim)*10,
                              dtype=tf.float32, trainable=learntop)
        self.logsigma = tf.Variable(tf.zeros(latent_dim),
                                    dtype=tf.float32, trainable=learntop)

        self.prior = tfd.MultivariateNormalDiag(
            self.mu, tf.math.exp(self.logsigma))



def train(num_epochs,
          batch_size,
          dataset_g,
          dataset_f,
          lr,
          exp_path,):


    # Print the experiment setup:
    print('Experiment setup:')
    print('---> num_epochs: {}'.format(num_epochs))
    print('---> batch_size: {}'.format(batch_size))
    print('---> dataset g : {}'.format(dataset_g))
    print('---> dataset f : {}'.format(dataset_f))
    print('---> Learning rate: {}'.format(lr))
    print('---> experiment path: {}'.format(exp_path))

    if os.path.exists(os.path.join(exp_path, 'logs')):
        shutil.rmtree(os.path.join(exp_path, 'logs'))

    ml_g_log_dir = os.path.join(exp_path, 'logs', 'ml_g')
    ml_g_summary_writer = tf.summary.create_file_writer(ml_g_log_dir)
    ml_g_loss_metric = tf.keras.metrics.Mean('ml_g_loss', dtype=tf.float32)

    ml_f_log_dir = os.path.join(exp_path, 'logs', 'ml_f')
    ml_f_summary_writer = tf.summary.create_file_writer(ml_f_log_dir)
    ml_f_loss_metric = tf.keras.metrics.Mean('ml_f_loss', dtype=tf.float32)
    
    mse_g_log_dir = os.path.join(exp_path, 'logs', 'mse_g')
    mse_g_summary_writer = tf.summary.create_file_writer(mse_g_log_dir)
    mse_g_loss_metric = tf.keras.metrics.Mean('mse_g_loss', dtype=tf.float32)

    mse_f_log_dir = os.path.join(exp_path, 'logs', 'mse_f')
    mse_f_summary_writer = tf.summary.create_file_writer(mse_f_log_dir)
    mse_f_loss_metric = tf.keras.metrics.Mean('mse_f_loss', dtype=tf.float32)

    cycle_log_dir = os.path.join(exp_path, 'logs', 'cycle')
    cycle_summary_writer = tf.summary.create_file_writer(cycle_log_dir)
    cycle_loss_metric = tf.keras.metrics.Mean('cycle_loss', dtype=tf.float32)
    
    
    train_dataset_g , test_dataset_g = Dataset_preprocessing(dataset=dataset_g ,batch_size = batch_size)
    train_dataset_f , test_dataset_f = Dataset_preprocessing(dataset=dataset_f ,batch_size = batch_size)
    
    test_dataset_g = next(iter(test_dataset_g))
    test_dataset_f = next(iter(test_dataset_f))

    print('Dataset g is loaded: training and test dataset shape: {} {}'.
          format(np.shape(next(iter(train_dataset_g))), np.shape(next(iter(test_dataset_g)))))

    print('Dataset f is loaded: training and test dataset shape: {} {}'.
          format(np.shape(next(iter(train_dataset_f))), np.shape(next(iter(test_dataset_f)))))

    _ , image_size , _ , c = np.shape(next(iter(train_dataset_g)))
    
    f = image_size//64

    f = 1 if f < 1 else f
    latent_dim = 4*f *4*f *4*c
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    

    pz_g = Prior(latent_dim = latent_dim)
    pz_f = Prior(latent_dim = latent_dim)
    model_g = generator(revnet_depth = model_depth ,
                      activation = inv_conv_activation,
                      f = f,
                      c = c,
                      image_size = image_size) # Injective network
    latent_model_g = latent_generator(network = 'injective',revnet_depth = latent_depth,
                                    f = f,
                                    c = c,
                                    image_size = image_size) # Bijective network

    model_f = generator(revnet_depth = model_depth ,
                      activation = inv_conv_activation,
                      f = f,
                      c = c,
                      image_size = image_size) # Injective network
    latent_model_f = latent_generator(network = 'injective',revnet_depth = latent_depth,
                                    f = f,
                                    c = c,
                                    image_size = image_size) # Bijective network
    
    dummy_x = next(iter(train_dataset_g))
    dummy_z, _ = model_f(dummy_x, reverse=False)
    dummy_l_z , _ = latent_model_f(dummy_z, reverse=False)
    dummy_z, _ = model_g(dummy_x, reverse=False)
    dummy_l_z , _ = latent_model_g(dummy_z, reverse=False)
    

    ckpt_g = tf.train.Checkpoint(pz= pz_g ,model=model_g, latent_model=latent_model_g)
    manager_g = tf.train.CheckpointManager(
        ckpt_g, os.path.join(exp_path_g, 'checkpoints'), max_to_keep=5)

    ckpt_g.restore(manager_g.latest_checkpoint)
    
    if manager_g.latest_checkpoint:
        print("Restored model g from {}".format(manager_g.latest_checkpoint))
        
        
       
    ckpt_f = tf.train.Checkpoint(pz = pz_f, model=model_f, latent_model=latent_model_f)
    manager_f = tf.train.CheckpointManager(
        ckpt_f, os.path.join(exp_path_f, 'checkpoints'), max_to_keep=5)

    ckpt_f.restore(manager_f.latest_checkpoint)
    
    if manager_f.latest_checkpoint:
        print("Restored model f from {}".format(manager_f.latest_checkpoint))
        
    
    generator_g = Unet(c)
    generator_f = Unet(c)
    # OUTPUT_CHANNELS = 1
    # generator_g = pix2pix_generator(OUTPUT_CHANNELS)
    # generator_f = pix2pix_generator(OUTPUT_CHANNELS)
    

    
    @tf.function
    def train_step_cycle(real_g, real_f):
        with tf.GradientTape() as tape:
            
            MAE = tf.keras.losses.MeanAbsoluteError()
            
            fake_g = generator_g(real_f)
            cycle_f = generator_f(fake_g)
            
            fake_f = generator_f(real_g)
            cycle_g = generator_g(fake_f)

            cycle_loss = MAE(real_g, cycle_g) + MAE(real_f, cycle_f)
            
            z_g , _ = model_g(fake_g, reverse= False)
            fake_g_hat , _ = model_g(z_g, reverse= True)
            mse_g = MAE(fake_g , fake_g_hat)
            
            z_f , _ = model_f(fake_f, reverse= False)
            fake_f_hat , _ = model_f(z_f, reverse= True)
            mse_f = MAE(fake_f , fake_f_hat)

            loss = mse_g + mse_f + cycle_loss 
            
        variables = []
        for v in generator_g.trainable_variables:
            variables.append(v)
        for v in generator_f.trainable_variables:
            variables.append(v)
        
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

        return mse_g , mse_f , cycle_loss
    
    @tf.function
    def train_step_cycleflow(real_g, real_f):
        with tf.GradientTape() as tape:
            
            MAE = tf.keras.losses.MeanAbsoluteError()
            
            fake_g = generator_g(real_f)
            cycle_f = generator_f(fake_g)
            
            fake_f = generator_f(real_g)
            cycle_g = generator_g(fake_f)


            #identity_f = generator_f(real_f)
            #identity_g = generator_g(real_g)

            #identity_loss_g = 5* MSE(real_g, identity_g)
            #identity_loss_f = 5* MSE(real_f, identity_f)

            cycle_loss = MAE(real_g, cycle_g) + MAE(real_f, cycle_f)
            
            z_g , _ = model_g(fake_g, reverse= False)
            fake_g_hat , _ = model_g(z_g, reverse= True)
            
            mse_g = MAE(fake_g , fake_g_hat)
            latent_sample_g, obj_g = latent_model_g(z_g, reverse=False)
            p_g = -tf.reduce_mean(pz_g.prior.log_prob(latent_sample_g))
            j_g = -tf.reduce_mean(obj_g)
            ml_loss_g =  p_g + j_g
            
            z_f , _ = model_f(fake_f, reverse= False)
            fake_f_hat , _ = model_f(z_f, reverse= True)
            
            mse_f = MAE(fake_f , fake_f_hat)
            
            latent_sample_f, obj_f = latent_model_f(z_f, reverse=False)
            p_f = -tf.reduce_mean(pz_f.prior.log_prob(latent_sample_f))
            j_f = -tf.reduce_mean(obj_f)
            ml_loss_f =  p_f + j_f
            
            loss = 100*mse_g + ml_loss_g + 100*mse_f + ml_loss_f + 100*cycle_loss 
            
        variables = []
        for v in generator_g.trainable_variables:
            variables.append(v)
        for v in generator_f.trainable_variables:
            variables.append(v)
        
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

        return ml_loss_g, ml_loss_f, mse_g , mse_f , cycle_loss

    
    time_vector = np.zeros([num_epochs,1])
                           
    for epoch in range(num_epochs):
        epoch_start = time()
        
        if epoch < ml_threshold:
            for x, y  in zip(train_dataset_g, train_dataset_f):
                mse_loss_g , mse_loss_f , cycle_loss = train_step_cycle(x,y)
                ml_loss_g = ml_loss_f = 0
            
        else:
            for x, y  in zip(train_dataset_g, train_dataset_f):
                
                ml_loss_g, ml_loss_f , mse_loss_g , mse_loss_f , cycle_loss = train_step_cycleflow(x,y)
        
        if epoch == 0:
            
            # Just for the first iteration of the first epoch
            # to calculate the number of trainable parametrs
            with tf.GradientTape() as tape:
                
                fake1 = generator_g(y)
                variables_gen_g = tape.watched_variables()
            
            with tf.GradientTape() as tape:
                
                fake2 = generator_f(x)
                variables_gen_f = tape.watched_variables()
                
            parameters_gen_g = np.sum([np.prod(v.get_shape().as_list()) for v in variables_gen_g])
            parameters_gen_f = np.sum([np.prod(v.get_shape().as_list()) for v in variables_gen_f])
            print('Number of trainable_parameters of generator g: {}'.format(parameters_gen_g))
            print('Number of trainable_parameters of generator f: {}'.format(parameters_gen_f))
            print('Total number of trainable_parameters: {}'.format(parameters_gen_g + parameters_gen_f))
    
       
        ml_g_loss_metric.update_state(ml_loss_g)
        ml_f_loss_metric.update_state(ml_loss_f)
        mse_g_loss_metric.update_state(mse_loss_g)
        mse_f_loss_metric.update_state(mse_loss_f)
        cycle_loss_metric.update_state(cycle_loss)
        
                           
        num_sample = 10
                           
        fake_f_test = generator_f(test_dataset_g[:num_sample])
        cycle_g_test = generator_g(fake_f_test)
        fake_g_test = generator_g(test_dataset_f[:num_sample])
        cycle_f_test = generator_f(fake_g_test)

        sample_g_f_g = tf.concat([test_dataset_g[:num_sample], fake_f_test, cycle_g_test], 0) 
        sample_f_g_f = tf.concat([test_dataset_f[:num_sample], fake_g_test, cycle_f_test], 0)
        
        sample_g_f_g = sample_g_f_g.numpy()
        sample_f_g_f = sample_f_g_f.numpy()
        
        samples_folder = os.path.join(exp_path, 'Generated_samples')
        
        if not os.path.exists(samples_folder):
            os.mkdir(samples_folder)



        if epoch == 0:

            z_g = pz_g.prior.sample(num_sample) # sampling from base (gaussian) with Temprature = 1
            z_g = latent_model_g(z_g , reverse = True)[0] # Intermediate samples with Temprature = 1
            x_g = model_g(z_g , reverse = True)[0] # Samples with Temprature = 1

            z_f = pz_f.prior.sample(num_sample) # sampling from base (gaussian) with Temprature = 1
            z_f = latent_model_f(z_f , reverse = True)[0] # Intermediate samples with Temprature = 1
            x_f = model_f(z_f , reverse = True)[0] # Samples with Temprature = 1

            x_total = tf.concat([x_g, x_f], 0)
            x_total = x_total.numpy()

            ngrid11 = 10
            ngrid22 = 2
                
            cv2.imwrite(os.path.join(samples_folder, 'flow_samples_epoch %d.png' % (epoch,)),
                        x_total[:, :, :, ::-1].reshape(
                ngrid22, ngrid11,
                image_size, image_size, c).swapaxes(1, 2)
                .reshape(ngrid22*image_size, -1, c)*127.5 + 127.5)
                           



        ngrid1 = 10
        ngrid2 = 3
                           
        cv2.imwrite(os.path.join(samples_folder, 'g-f-g_epoch %d.png' % (epoch,)),
                    sample_g_f_g[:, :, :, ::-1].reshape(
            ngrid2, ngrid1,
            image_size, image_size, c).swapaxes(1, 2)
            .reshape(ngrid2*image_size, -1, c)*127.5 + 127.5)
                           
                           
        
        cv2.imwrite(os.path.join(samples_folder, 'f-g-f_epoch %d.png' % (epoch,)),
                    sample_f_g_f[:, :, :, ::-1].reshape(
            ngrid2, ngrid1,
            image_size, image_size, c).swapaxes(1, 2)
            .reshape(ngrid2*image_size, -1, c)*127.5 + 127.5)
                           
            
        with ml_g_summary_writer.as_default():
            tf.summary.scalar(
                'ml_g_loss', ml_g_loss_metric.result(), step=epoch)

        with ml_f_summary_writer.as_default():
            tf.summary.scalar(
                'ml_f_loss', ml_f_loss_metric.result(), step=epoch)

        with mse_g_summary_writer.as_default():
            tf.summary.scalar(
                'mse_g_loss', mse_g_loss_metric.result(), step=epoch)

        with mse_f_summary_writer.as_default():
            tf.summary.scalar(
                'mse_f_loss', mse_f_loss_metric.result(), step=epoch)

        with cycle_summary_writer.as_default():
            tf.summary.scalar(
                'cycle_loss', cycle_loss_metric.result(), step=epoch)
            
            
        print("Epoch {:03d}: ml_g loss: {:.3f} / ml_f loss: {:.3f} / mse_g loss: {:.3f} / mse_f loss: {:.3f} / cycle loss: {:.3f} "
              .format(epoch, ml_g_loss_metric.result().numpy(), ml_f_loss_metric.result().numpy(),
                      mse_g_loss_metric.result().numpy(), mse_f_loss_metric.result().numpy() ,cycle_loss_metric.result().numpy()))
        
        

        ml_g_loss_metric.reset_states()
        ml_f_loss_metric.reset_states()
        mse_g_loss_metric.reset_states()
        mse_f_loss_metric.reset_states()
        cycle_loss_metric.reset_states()
        
        epoch_end = time()
        time_vector[epoch] = epoch_end - epoch_start
        np.save(os.path.join(exp_path, 'time_vector.npy') , time_vector)
        print('epoch time:{}'.format(time_vector[epoch]))

   
        
if __name__ == '__main__':
    train(num_epochs,
          batch_size,
          dataset_g,
          dataset_f,
          lr,
          exp_path)