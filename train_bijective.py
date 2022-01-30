from my_models import generator, latent_generator
import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import cv2
from my_utils import *
import os
import shutil
from time import time


tfb = tfp.bijectors
tfd = tfp.distributions

FLAGS, unparsed = flags()

num_epochs = FLAGS.num_epochs
batch_size = FLAGS.batch_size
dataset = FLAGS.dataset
lr = FLAGS.lr
gpu_num = FLAGS.gpu_num
learntop = bool(FLAGS.learntop)
remove_all = bool(FLAGS.remove_all)
desc = FLAGS.desc
ml_threshold = FLAGS.ml_threshold
model_depth = FLAGS.model_depth
latent_depth = FLAGS.latent_depth
inv_conv_activation = FLAGS.inv_conv_activation
T = FLAGS.T


all_experiments = 'experiment_results/'
if os.path.exists(all_experiments) == False:

    os.mkdir(all_experiments)

# experiment path
exp_path = all_experiments + 'Bijective_' + \
    dataset + '_' + 'model_depth_%d' % (model_depth,) + '_' + 'latent_depth_%d'% (latent_depth,) + '_learntop_%d' \
        % (int(learntop)) + '_' + desc


if os.path.exists(exp_path) == True and remove_all == True:
    shutil.rmtree(exp_path)

if os.path.exists(exp_path) == False:
    os.mkdir(exp_path)


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
            
        self.mu = tf.Variable(tf.zeros(latent_dim),
                              dtype=tf.float32, trainable=learntop)
        self.logsigma = tf.Variable(tf.zeros(latent_dim),
                                    dtype=tf.float32, trainable=learntop)

        self.prior = tfd.MultivariateNormalDiag(
            self.mu, tf.math.exp(self.logsigma))



def latent_space_interplotion(model, x1, x2, latent=True , sample_number = 16):
    """Creates a grid of images from x1 to x2"""
    if not latent:
        """if latent then x1 and x2 are treated to be latent codes"""
        z1, _ = model(x1, reverse=True)
        z2, _ = model(x2, reverse=True)
    else:
        z1 = x1
        z2 = x2

    # create a grid of latent codes
    
    a = tf.cast(tf.reshape(tf.linspace(0, 1, sample_number), (sample_number, 1)), tf.float32)
    z = z1 + a * (z2 - z1)
    xhat = model(z, reverse= True)[0]

    return xhat.numpy()


def train(num_epochs,
          batch_size,
          dataset,
          lr,
          exp_path,):


    # Print the experiment setup:
    print('Experiment setup:')
    print('---> num_epochs: {}'.format(num_epochs))
    print('---> batch_size: {}'.format(batch_size))
    print('---> dataset: {}'.format(dataset))
    print('---> Learning rate: {}'.format(lr))
    print('---> experiment path: {}'.format(exp_path))
    
    if os.path.exists(os.path.join(exp_path, 'logs')):
        shutil.rmtree(os.path.join(exp_path, 'logs'))


    ML_log_dir = os.path.join(exp_path, 'logs', 'ML')
    ML_summary_writer = tf.summary.create_file_writer(ML_log_dir)
    ML_loss_metric = tf.keras.metrics.Mean('ML_loss', dtype=tf.float32)
    
    pz_log_dir = os.path.join(exp_path, 'logs', 'pz')
    pz_summary_writer = tf.summary.create_file_writer(pz_log_dir)
    pz_metric = tf.keras.metrics.Mean(
        'pz', dtype=tf.float32)
    
    jacobian_log_dir = os.path.join(exp_path, 'logs', 'jacobian')
    jacobian_summary_writer = tf.summary.create_file_writer(jacobian_log_dir)
    jacobian_metric = tf.keras.metrics.Mean(
        'jacobian', dtype=tf.float32)
    
    

    train_dataset , test_dataset = Dataset_preprocessing(dataset=dataset ,batch_size = batch_size)
    
    print('Dataset is loaded: training and test dataset shape: {} {}'.
          format(np.shape(next(iter(train_dataset))), np.shape(next(iter(test_dataset)))))

    _ , image_size , _ , c = np.shape(next(iter(train_dataset)))
    
    latent_dim = image_size * image_size * c

    
    initial_learning_rate = lr
    # steps = num_epochs * ( 60000//batch_size)
    # print(lr)
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    # initial_learning_rate,
    # decay_steps=steps//2,
    # decay_rate=0.3,
    # staircase=True)

    
    f_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    pz = Prior(latent_dim = latent_dim)
    
    time_vector = np.zeros([num_epochs,1]) # time per epoch
   
    latent_model = latent_generator(network = 'bijective',revnet_depth = latent_depth,
                                    c = c,
                                    image_size = image_size) # Bijective network


     # call generator once to set weights (Data dependent initialization)
    dummy_x = next(iter(train_dataset))
    dummy_z , _ = latent_model(dummy_x, reverse=False)
    

    ckpt = tf.train.Checkpoint(pz = pz,latent_model=latent_model,f_optimizer=f_optimizer)
    manager = tf.train.CheckpointManager(
        ckpt, os.path.join(exp_path, 'checkpoints'), max_to_keep=5)

    ckpt.restore(manager.latest_checkpoint)
    
    
    @tf.function
    def train_step_ml(sample):
        """ML training of the Injective network"""

        with tf.GradientTape() as tape:
            latent_sample, obj = latent_model(sample, reverse=False)
            p = -tf.reduce_mean(pz.prior.log_prob(latent_sample))
            j = -tf.reduce_mean(obj) # Log-det of Jacobian
            loss =  p + j
            variables = tape.watched_variables()
            grads = tape.gradient(loss, variables)
            f_optimizer.apply_gradients(zip(grads, variables))

        return loss , p , j

   

    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    
    for epoch in range(num_epochs):
        
        epoch_start = time()
        
        # ML training of the bijective network after ml threshold epochs
        for x in train_dataset:
        
            ml_loss , p , j = train_step_ml(x)
            
        if epoch == 0:
        
            # Just for the first iteration of the first epoch
            # to calculate the number of trainable parametrs
            
            with tf.GradientTape() as tape:
                
                _, _ = latent_model(x, reverse=False)
                variables_latent_model = tape.watched_variables()
                
            
            parameters_latent_model = np.sum([np.prod(v.get_shape().as_list()) for v in variables_latent_model])
            print('Number of trainable_parameters of bijective model: {}'.format(parameters_latent_model))
         
    
        ML_loss_metric.update_state(ml_loss)
        pz_metric.update_state(p)
        jacobian_metric.update_state(j)
        
        sample_number = 25 # Number of samples to show
        
      
        # Sampling from distribution
        z_random_base = pz.prior.sample(sample_number) # sampling from base (gaussian) with Temprature = 1
        z_random_base_T = (z_random_base - pz.mu) * T + pz.mu # sampling from base (gaussian) with Temprature = T
        x_sampled = latent_model(z_random_base , reverse = True)[0].numpy() # Intermediate samples with Temprature = 1
        x_sampled_T = latent_model(z_random_base_T , reverse = True)[0].numpy() # Intermediate samples with Temprature = T
        
        
        # Saving experiment results
        samples_folder = os.path.join(exp_path, 'Generated_samples')
        if not os.path.exists(samples_folder):
            os.mkdir(samples_folder)
        
        ngrid = int(np.sqrt(sample_number))


        image_path_sampled = os.path.join(samples_folder, 'sampled')
        if os.path.exists(image_path_sampled) == False:
            os.mkdir(image_path_sampled)

        cv2.imwrite(os.path.join(image_path_sampled, 'sampled_epoch %d.png' % (epoch,)),
                    x_sampled[:, :, :, ::-1].reshape(
            ngrid, ngrid,
            image_size, image_size, c).swapaxes(1, 2)
            .reshape(ngrid*image_size, -1, c)*127.5 + 127.5) # samples from distribution with Temprature = 1
        
        cv2.imwrite(os.path.join(image_path_sampled, 'Tempreture_sampled_epoch %d.png' % (epoch,)),
                    x_sampled_T[:, :, :, ::-1].reshape(
            ngrid, ngrid,
            image_size, image_size, c).swapaxes(1, 2)
            .reshape(ngrid*image_size, -1, c)*127.5 + 127.5) # samples from distribution with Temprature = T
    
        
        # Saving logs
        with ML_summary_writer.as_default():
            tf.summary.scalar(
                'ML_loss', ML_loss_metric.result(), step=epoch)

        
        with pz_summary_writer.as_default():
            tf.summary.scalar(
                'pz', pz_metric.result(), step=epoch)
            
        
        with jacobian_summary_writer.as_default():
            tf.summary.scalar(
                'jacobian', jacobian_metric.result(), step=epoch)
            
        
        print("Epoch {:03d}:  ML Loss: {:.3f} "
              .format(epoch, ML_loss_metric.result().numpy()))
        
        
        ML_loss_metric.reset_states()
        pz_metric.reset_states()
        jacobian_metric.reset_states()

        save_path = manager.save()
        print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))
        
        epoch_end = time()
        time_vector[epoch] = epoch_end - epoch_start
        np.save(os.path.join(exp_path, 'time_vector.npy') , time_vector)
        print('epoch time:{}'.format(time_vector[epoch]))
        



if __name__ == '__main__':
    train(num_epochs,
          batch_size,
          dataset,
          lr,
          exp_path)
