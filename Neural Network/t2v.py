from keras import backend as K
from keras.layers import *
from keras.models import *
from keras.callbacks import *

class T2V(Layer):

    def __init__(self, k=None, **kwargs):
        self.k = k
        super(T2V, self).__init__(**kwargs)

    def build(self, input_shape):
        self.timesteps = input_shape[1] 
        self.features = input_shape[2]
        
        self.W = self.add_weight(name='W', 
                              shape=(self.timesteps, self.k*self.features),
                              initializer='uniform',
                              trainable=True)
        
        self.B = self.add_weight(name='B', 
                              shape=(self.timesteps, self.k*self.features),
                              initializer='uniform',
                              trainable=True)
        
        self.A = self.add_weight(name='A', 
                              shape=(self.timesteps, self.k*self.features),
                              initializer='uniform',
                              trainable=True)
            
        self.w = self.add_weight(name='w', 
                              shape=(1, self.features),
                              initializer='uniform',
                              trainable=True)
        
        self.b = self.add_weight(name='b', 
                              shape=(self.timesteps, self.features),
                              initializer='uniform',
                              trainable=True)

        super(T2V, self).build(input_shape) 

    def call(self, x):
        
        linear_term = self.w * x + self.b     
        
        x = K.repeat_elements(x, rep=self.k, axis=-1)   
                                                        
       
        sin_trans = K.sin(x * self.W + self.B) * self.A
        
        return  K.concatenate([sin_trans, linear_term], axis=-1)
        

    def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[1], self.k*self.features+self.features) 
