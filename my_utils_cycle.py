import tensorflow as tf
import numpy as np
import cv2
import argparse
from sklearn.utils import shuffle


snr = 10

def generate_sigma(target):
    return 10 ** (-snr / 20.0) * np.sqrt(np.mean(np.sum(np.square(np.reshape(target, (np.shape(target)[0], -1))), -1)))


def denoise(target):
    
    noise_sigma = generate_sigma(target)
    noise = np.random.normal(loc=0, scale=noise_sigma, size=np.shape(target))/np.sqrt(np.prod(np.shape(target)[1:]))
    noisy = target + noise
    return noisy

def data_normalization(x):
    
    x = x.astype('float32')
    x = x - (x.max() + x.min())/2
    x /= (x.max())
    
    return x


def Dataset_preprocessing(dataset = 'MNIST', batch_size = 64):
    
    if dataset == 'mnist':
        
        nch = 1
        r = 32
        (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
        
    
    elif dataset == 'noisy_mnist':
    
        (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

        r = 32
        nch = 1


    elif dataset == 'fmnist':
    
        (train_images, _), (test_images, _) = tf.keras.datasets.fashion_mnist.load_data()
        r = 32
        nch = 1

    elif dataset == 'cifar10':
        (train_images, _), (test_images, _) = tf.keras.datasets.cifar10.load_data()
        r = 32
        nch = 3
        
    elif dataset == 'svhn':
        
        train_images, test_images = svhn()
        nch = 3
        r = 32

    elif dataset == 'celeba':
        celeba = np.load('/kaggle/working/celeb.npy')
        celeba = shuffle(celeba)
        train_images, test_images = np.split(celeba, [80000], axis=0)
        nch = 3
        r = 32
        
    elif dataset == 'imagenet':
        imagenet = np.load('/raid/Amir/Projects/datasets/Tiny_imagenet.npy')
        imagenet = shuffle(imagenet)
        train_images, test_images = np.split(imagenet, [80000], axis=0)
        nch = 3
        r = 64
        
    elif dataset == 'rheo':
        rheo = np.load('/raid/Amir/Projects/datasets/rheology.npy')
        rheo = shuffle(rheo)
        train_images, test_images = np.split(rheo, [1500], axis=0)
        nch = 3
        r = 64
        
        
    elif dataset == 'chest':
        chest = np.load('/raid/Amir/Projects/datasets/X_ray_dataset_128.npy')[:100000,:,:,0:1]
        chest = shuffle(chest)
        print(np.shape(chest))
        train_images, test_images = np.split(chest, [80000], axis=0)
        # print(type(train_images[0,0,0,0]))
        nch = 1
        r = 128
    
    
    elif dataset == 'church':
        church = np.load('/raid/Amir/Projects/datasets/church_outdoor_train_lmdb_color_64.npy')[:100000,:,:,:]
        church = shuffle(church)
        print(np.shape(church))
        train_images, test_images = np.split(church, [80000], axis=0)
        # print(type(train_images[0,0,0,0]))
        nch = 3
        r = 64
        
        

    training_images = np.zeros((np.shape(train_images)[0], r, r, nch))
    testing_images = np.zeros((np.shape(test_images)[0], r, r, nch))

    if train_images.shape[1] != r:

        for i in range(np.shape(train_images)[0]):
            if nch == 1:
                training_images[i,:,:,0] = cv2.resize(train_images[i] , (r,r))
            else:
                training_images[i] = cv2.resize(train_images[i] , (r,r))

        for i in range(np.shape(test_images)[0]):
            if nch == 1:
                testing_images[i,:,:,0] = cv2.resize(test_images[i] , (r,r))
            else:
                testing_images[i] = cv2.resize(test_images[i] , (r,r))

    else:
        training_images = train_images
        testing_images = test_images

    # Normalize the images to [-1, 1]
    
    # training_images = training_images[0:200]

    if dataset == 'noisy_mnist':
        
        training_images = denoise(training_images)
        testing_images = denoise(testing_images)
    
    training_images = data_normalization(training_images)
    testing_images = data_normalization(testing_images)
    
    training_images = tf.convert_to_tensor(training_images, tf.float32)
    testing_images = tf.convert_to_tensor(testing_images, tf.float32)
   
    train_dataset = tf.data.Dataset.from_tensor_slices((training_images))
    test_dataset = tf.data.Dataset.from_tensor_slices((testing_images))
    
    SHUFFLE_BUFFER_SIZE = 256
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size,
                                                                     drop_remainder = True).prefetch(5)
    test_dataset = test_dataset.batch(batch_size)

    
    return train_dataset , test_dataset
      


def flags():

    parser = argparse.ArgumentParser()
     
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        help='number of epochs to train for')
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='batch_size')

   
    parser.add_argument(
        '--dataset_g', 
        type=str,
        default='mnist',
        help='which dataset to work with')
    
    
    parser.add_argument(
        '--dataset_f', 
        type=str,
        default='noisy_mnist',
        help='which dataset to work with')
    
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='learning rate')
    
    parser.add_argument(
        '--model_depth',
        type=int,
        default= 2,
        help='revnet depth of model')
    
    parser.add_argument(
        '--latent_depth',
        type=int,
        default= 3,
        help='revnet depth of latent model')
    
    
    parser.add_argument(
        '--learntop',
        type=int,
        default=1,
        help='Trainable top')
    
    parser.add_argument(
        '--gpu_num',
        type=int,
        default=0,
        help='GPU number')

    parser.add_argument(
        '--remove_all',
        type= int,
        default= 1,
        help='Remove the previous experiment')
    
    parser.add_argument(
        '--desc',
        type=str,
        default='cycle',
        help='add a small descriptor to folder name')
    
    parser.add_argument(
        '--inv_conv_activation',
        type=str,
        default= 'linear',
        help='activation of invertible 1x1 conv layer')
    
    parser.add_argument(
        '--T',
        type=float,
        default= 1,
        help='sampling tempreture')
    
    
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed

