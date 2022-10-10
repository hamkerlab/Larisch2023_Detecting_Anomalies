import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import os

########
# Compact Convolutional Transfomer 
# 
# Definition if the CCT, based on the implementation of Sayak Paul for keras (https://keras.io/examples/vision/cct/)
# The CCT is slightly modified for the anomaly prediction
#######




### set the CUDA_VISIBLE_DEVICES variable to -1 if you want to use the CPU
### selse, use the commented code below to determine the GPU-ID you want to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

"""
gpus = tf.config.list_physical_devices('GPU')
print('GPUS: ', gpus)
if gpus:
  # Restrict TensorFlow to only use one GPU
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
"""

#####
## Convolutional Tokenizer for the convolutional - embedding
#####
class CCTTokenizer(tf.keras.layers.Layer):
    def __init__(
        self,
        kernel_size=4,
        stride = 1,
        padding=1,
        pooling_kernel_size=2,
        pooling_stride = 2,
        num_conv_layers = 2,
        num_output_channels =[128, 64],
        positional_emb = None,
        **kwargs,):
        
        super(CCTTokenizer, self).__init__(**kwargs)



        ## Tokenizer
        self.conv_model = tf.keras.Sequential()
        for i in range(num_conv_layers):
            self.conv_model.add(
                tf.keras.layers.Conv2D(
                    num_output_channels[i],
                    kernel_size,
                    stride,
                    padding="valid",
                    use_bias = False,
                    activation="relu",
                    kernel_initializer="he_normal",)
            )
            self.conv_model.add(tf.keras.layers.ZeroPadding2D(padding))    
            self.conv_model.add(tf.keras.layers.MaxPool2D(pooling_kernel_size, pooling_stride,"same"))
            
        self.positional_emb = positional_emb


    def call(self, images):
        outputs = self.conv_model(images)
        # After passing the images through our mini-network the spatial dimensions
        # are flattened to form sequences
        reshaped = tf.reshape(outputs,(-1, tf.shape(outputs)[1] * tf.shape(outputs)[2], tf.shape(outputs)[-1]),)
        return reshaped


    #NOTE1: Without Embedding there is a None-Type error
    #NOTE2: the dummy inputs is in the original version quadratic/ -> change to input_dim
    def positional_embedding(self, input_shape):
        # Positional embeddings are optional in CCT. Here, we calculate 
        # the number of sequences and initialize an "Embedding" layer to
        # compute the positional embeddings later 
        if self.positional_emb:
            dummy_inputs = tf.ones((1, input_shape[0], input_shape[1],input_shape[2]))
            dummy_outputs = self.call(dummy_inputs)
            sequence_length = tf.shape(dummy_outputs)[1]
            projection_dim = tf.shape(dummy_outputs)[-1]
            embed_layer = tf.keras.layers.Embedding(input_dim = sequence_length, output_dim = projection_dim)
            return embed_layer, sequence_length
        else:
            return None


## Stochastic depth as a regularization method.
# Like Dropout, but for blocks of layers
# See: github.com/rwightman/pytorch-image-models
class StochasticDepth(tf.keras.layers.Layer):
    def __init__(self, drop_pop, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_pop = drop_pop
    
    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_pop
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x/ keep_prob) * random_tensor
        return x

## MLP for the Transformers encoder
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x

### The Final CCT Model ####
def create_cct_model(image_size, input_shape, num_classes, num_heads, projection_dim,num_conv_layers, num_output_channels, transformer_units,transformer_layers, positional_emb):


    ## Using a GaussianNoise-Layer to perform some data augmentation
    data_augmentation = tf.keras.layers.GaussianNoise(1.0)

    # Encode patches
    cct_tokenizer = CCTTokenizer(positional_emb=positional_emb, num_conv_layers =num_conv_layers , num_output_channels = num_output_channels)

    ## get the lenght of the window
    l_window = input_shape[1]

    ## Build the model with functional API
    inputs = tf.keras.layers.Input(input_shape)
    augmented = data_augmentation(inputs)
    encoded_patches = cct_tokenizer(augmented)


    # Apply positional embedding
    if positional_emb:
        pos_embed, seq_length = cct_tokenizer.positional_embedding(input_shape)
        positions = tf.range(start=0, limit=seq_length, delta=1)
        position_embeddings = pos_embed(positions)
        encoded_patches += position_embeddings

    stochastic_depth_rate = 0.25

    # Calculate Stochastic Depth probabilities
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block (!)
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

        # Create a multi-head (self-) attention layer
        attention_output,attention_scores = tf.keras.layers.MultiHeadAttention(
                            num_heads = num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1,return_attention_scores=True)
        
        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x2) #, center=False, scale=False

        # MLP
        x3 = mlp(x3, hidden_units = transformer_units, dropout_rate=0.1)

        # Skip connection 2
        x3 =  StochasticDepth(dpr[i])(x3)
        encoded_patches = tf.keras.layers.Add()([x3,x2])

    # Apply sequence pooling
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-5)(encoded_patches) #, center=False, scale=False
    attention_weights = tf.nn.softmax(tf.keras.layers.Dense(1)(representation), axis=1)
    weighted_representation = tf.matmul(attention_weights, representation, transpose_a = True)
    weighted_representation = tf.squeeze(weighted_representation, -2)
    

    # predict the next values as a regression problem
    dense = tf.keras.layers.Dense(l_window*60)(weighted_representation)
    dense = tf.keras.layers.Reshape((l_window,60))(dense)
    logits = tf.keras.layers.Dense(num_classes, activation='softmax')(dense)


    # Create the Keras model.
    model = tf.keras.Model(inputs = inputs, outputs = logits)

    return(model)

