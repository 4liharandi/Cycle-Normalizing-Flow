import tensorflow as tf
import numpy as np
from my_utils import *
import glow_ops as g


class generator(tf.keras.Model):
    def __init__(self, **kwargs):
        super(generator, self).__init__()
        """Bijective Model architecture
        . upsqueeze
        --> revnet
        |-> inj_rev_step

        + 4x4x12 --> 4x4x12 |-> 4x4x24 . 8x8x6
         --> 8x8x6 |-> 8x8x12 |-> 8x8x24 --> 8x8x24
        |-> 8x8x48 --> 8x8x48 . 16x16x12 |-> 16x16x24
        --> 16x16x24 . 32x32x6 |-> 32x32x12 --> 32x32x12
        . 64x64x3
        
        summary for celeba: 
        6 bijective revnets
        6 injective revnet_steps
        4 upsqueeze
        """
        self.depth = kwargs.get('revnet_depth', 3) # revnet depth
        self.activation = kwargs.get('activation', 'linear') # activation ofinvertible 1x1 convolutional layer
        self.f = kwargs.get('f', 1)
        self.c = kwargs.get('c', 3)
        self.image_size = kwargs.get('image_size', 32)
        
        self.squeeze = g.upsqueeze(factor=2)
        self.revnets = [g.revnet(depth= self.depth , latent_model = False) 
        for _ in range(7)] # Bijective revnets
        
        self.inj_rev_steps = [g.revnet_step(layer_type='injective',
            latent_model = False, activation = self.activation) for _ in range(7)]
        

    def call(self, x, reverse=False , training = True):
        
        
        
        if reverse:
                x = tf.reshape(x, [-1,4*self.f, 4*self.f, 4*self.c])
                
        ops = [
        self.squeeze,
        self.revnets[0],
        self.inj_rev_steps[0],
        self.revnets[1],
        self.squeeze,
        self.revnets[2],
        self.inj_rev_steps[1],
        self.revnets[3],
        self.squeeze,
        self.inj_rev_steps[2],
        self.revnets[4],
        self.inj_rev_steps[4],
        ]
        
        
        if self.image_size >= 64:
            
            ops += [self.inj_rev_steps[5],
            self.revnets[5],
            self.squeeze,
            self.inj_rev_steps[6],
            self.revnets[6]
            ]
   

        if reverse:
            ops = ops[::-1]

        objective = 0.0

        for op in ops:
            
            x, curr_obj = op(x, reverse= reverse , training = training)
            objective += curr_obj


        if not reverse:
            x = tf.reshape(x, (-1, 4*self.f *4*self.f *4*self.c))

        return x, objective
 


class latent_generator(tf.keras.Model):
    def __init__(self, **kwargs):
        super(latent_generator, self).__init__()
        """ Injective Model architecture
        --> revnet
        
        + 4x4x12 --> 4x4x12 --> 4x4x12 --> 4x4x12 -->
        4x4x12 --> 4x4x12 --> 4x4x12 --> 4x4x12 -->
        4x4x12 --> 4x4x12 --> 4x4x12 --> 4x4x12 -->
        4x4x12 --> 4x4x12 --> 4x4x12 --> 4x4x12 -->
        4x4x12
        
        summary for celeba: 
        8 bijective revnets
        """
        self.network = kwargs.get('network', 'injective') # revnet depth
        self.depth = kwargs.get('revnet_depth', 3) # revnet depth
        self.f = kwargs.get('f', 1)
        self.c = kwargs.get('c', 3)
        self.squeeze = g.upsqueeze(factor=2)
        self.image_size = kwargs.get('image_size', 32)
        self.depth = kwargs.get('revnet_depth', 3) # revnet depth
        self.revnets = [g.revnet(coupling_type='affine', depth = self.depth , latent_model = True) 
        for _ in range(6)]
    def call(self, x, reverse=False , training = True):
        
        if self.network == 'injective':
    
            x = tf.reshape(x, [-1,4*self.f, 4*self.f, 4*self.c])
            
            ops = [
            self.revnets[0],
            self.revnets[1],
            self.revnets[2],
            self.revnets[3],
            self.revnets[4],
            self.revnets[5]
            ]
        
        else:
            
            if reverse:
                x = tf.reshape(x, [-1,4,4, (self.image_size//4) **2  * self.c])
            
            ops = [
            self.squeeze,
            self.revnets[0],
            self.squeeze,
            self.revnets[1],
            self.squeeze,
            self.revnets[2],
            self.revnets[3]]
        
            if self.image_size == 64:
                ops = ops + [self.squeeze,
                             self.revnets[4],
                             self.revnets[5]]

        if reverse:
            ops = ops[::-1]

        objective = 0.0

        for op in ops:
            
            x, curr_obj = op(x, reverse=reverse , training = training)

            objective += curr_obj


        if self.network == 'injective':
                x = tf.reshape(x, (-1, 4*self.f *4*self.f *4*self.c))
        else:
            if not reverse:
                x = tf.reshape(x, (-1, self.image_size*self.image_size*self.c))
                
                
        return x, objective
 



def unit_test_generator():
    
    MSE = tf.keras.losses.MeanSquaredError()
    
    x = tf.random.normal(shape = [100 , 32 , 32 , 1])
    z = tf.random.normal(shape = [100 , 4*4*4])
    
    g = generator(f = 1, c = 1 , image_size = 4)
    x_int , _ = g(x , reverse = False)
    print(np.shape(x_int))
    z_int , _ = g(z , reverse = True)
    print(np.shape(z_int))
    z_hat , _ = g(z_int , reverse = False)
    print(np.max(z_int))
    loss = MSE(z , z_hat)
    
    print(loss)
    
    
if __name__ == '__main__':
    unit_test_generator()
    # unit_test_latent_generator()