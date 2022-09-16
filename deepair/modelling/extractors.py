import os
from builtins import input
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow.keras as keras

#%%
def conv_seq_extractor(hp, seq_len, vocab_size, name=None):
    """ convolutional network for sequences
    
    The sequence input will initially be embedded by a trainable 
    embedding layer.
    
    Multiple 1D convolutions along a sequence can be applied, followed
    by a global max pool operation at the last convolutional output.
    
    Convolutions can be standard or dilated.
    
    parameters
    -----------
    
    hp: dict
        A dictionary of hyperparameters for the model:
         - embed_dim: int
                 dimension to embed each element of the sequences
         - filters: list
                 list of ints, element i of the list is the number of 
                 filters to put in the i'th conv layer
        - strides: list
                 list of ints, element i of the list is the strides to 
                 use in the i'th conv layer. strides[i] must be 1 where
                 dilations[i]=/=1
        - kernel_widths: list
                 list of ints, element i of the list is the width of the  
                 kernel to use in the filter of the i'th conv layer
        - dilations: list
                list of ints for the dilation of conv layer i.
        - L2_conv: float
                value for L2 norm penalty for each conv layer
        - dropout_conv: float
                dropout to apply following each conv layer
                
    returns
    --------
    model: tf.keras.Model
            Keras model that converts input into a a feature representation. i.e prior
            to any final dense layers.
    
    
    """
    model_in = keras.Input(shape = (seq_len,))
    
    embedder = keras.layers.Embedding(vocab_size,hp['embed_dim'])
    
    convs=[]
    bns = []
    
    strides = [1]*len(hp['filters'])
    if hp['strides']:
        strides=hp['strides']
    
    for f,k,d,s in zip(hp['filters'],hp['kernel_widths'],hp['dilations'],strides):
        convs.append( keras.layers.Conv1D(f,
                                  kernel_size=k,
                                  dilation_rate=d,
                                  strides=s,
                                  kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=hp['L2_conv']),
                                  padding='same')
                    )
        bns.append( keras.layers.BatchNormalization() )
                    
    
    elu_activation = keras.layers.ELU()
    
    dropout_conv = keras.layers.Dropout(hp['dropout_conv'])
    
    x = embedder(model_in)
    
    for c,bn in zip(convs,bns):
        x = c(x)
        x = elu_activation(x)
        x = dropout_conv(x)
        x = bn(x)
        
    x = keras.layers.GlobalMaxPool1D()(x)
    
    return keras.Model(inputs=model_in, outputs=x, name=name)

def conv_seq_transformer_extractor(seq_transformer_info, 
                                  hp, 
                                  transformer_feature='last_hidden_state', 
                                  name=None
    ):
    """ convolutional network for sequences
    
    The sequence input will initially be embedded by a trainable 
    embedding layer.
    
    Multiple 1D convolutions along a sequence can be applied, followed
    by a global max pool operation at the last convolutional output.
    
    Convolutions can be standard or dilated.
    
    parameters
    -----------
    
    hp: dict
        A dictionary of hyperparameters for the model:
         - embed_dim: int
                 dimension to embed each element of the sequences
         - filters: list
                 list of ints, element i of the list is the number of 
                 filters to put in the i'th conv layer
        - strides: list
                 list of ints, element i of the list is the strides to 
                 use in the i'th conv layer. strides[i] must be 1 where
                 dilations[i]=/=1
        - kernel_widths: list
                 list of ints, element i of the list is the width of the  
                 kernel to use in the filter of the i'th conv layer
        - dilations: list
                list of ints for the dilation of conv layer i.
        - L2_conv: float
                value for L2 norm penalty for each conv layer
        - dropout_conv: float
                dropout to apply following each conv layer
                
    returns
    --------
    model: tf.keras.Model
            Keras model that converts input into a a feature representation. i.e prior
            to any final dense layers.
    
    
    """

    tokenizer_fea_len = seq_transformer_info['tokenizer_fea_len']
    SeqInforModel = seq_transformer_info['SeqInforModel']
    
    model_in = keras.Input(shape = (tokenizer_fea_len,),dtype='int64')

    if transformer_feature == 'last_hidden_state':
        convs=[]
        bns = []
        
        strides = [1]*len(hp['filters'])
        if hp['strides']:
            strides=hp['strides']
        
        for f,k,d,s in zip(hp['filters'],hp['kernel_widths'],hp['dilations'],strides):
            convs.append(keras.layers.Conv1D(f,
                                    kernel_size=k,
                                    dilation_rate=d,
                                    strides=s,
                                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=hp['L2_conv']),
                                    padding='same')
                        )
            bns.append(keras.layers.BatchNormalization())                    
        
        elu_activation = keras.layers.ELU()
        
        dropout_conv = keras.layers.Dropout(hp['dropout_conv'])

    # print('-'*30)
    # print(model_in)
    # print('-'*30)
   
    x = SeqInforModel.bert(model_in)
    
    # print('-'*30)
    # print(x)
    # print('-'*30)
    
    if transformer_feature == 'last_hidden_state':
        x = x[transformer_feature] 
        # x = keras.layers.BatchNormalization()(x)
        for c,bn in zip(convs,bns):
            x = c(x)
            x = elu_activation(x)
            x = dropout_conv(x)
            x = bn(x)        
        x = keras.layers.GlobalMaxPool1D()(x)
    elif transformer_feature == 'pooler_output':
        x = x[transformer_feature] 
        # print('-'*30)
        # print(x)
        # print('-'*30)
    else:
        raise RuntimeError('Check transformer output!')
    
    return keras.Model(inputs=model_in, outputs=x, name=name)

def conv_seq_transformer_extractor_withpretrain(seq_transformer_info, 
                                                hp, 
                                                name=None
    ):
    """ 
    convolutional network for sequences
    
    The sequence input will initially be embedded by a trainable 
    embedding layer.
    
    Multiple 1D convolutions along a sequence can be applied, followed
    by a global max pool operation at the last convolutional output.
    
    Convolutions can be standard or dilated.
    
    parameters
    -----------    
    hp: dict
        A dictionary of hyperparameters for the model:
         - embed_dim: int
                 dimension to embed each element of the sequences
         - filters: list
                 list of ints, element i of the list is the number of 
                 filters to put in the i'th conv layer
        - strides: list
                 list of ints, element i of the list is the strides to 
                 use in the i'th conv layer. strides[i] must be 1 where
                 dilations[i]=/=1
        - kernel_widths: list
                 list of ints, element i of the list is the width of the  
                 kernel to use in the filter of the i'th conv layer
        - dilations: list
                list of ints for the dilation of conv layer i.
        - L2_conv: float
                value for L2 norm penalty for each conv layer
        - dropout_conv: float
                dropout to apply following each conv layer
                
    returns
    --------
    model: tf.keras.Model
            Keras model that converts input into a a feature representation. i.e prior
            to any final dense layers. 
    """
   
    model_in = keras.Input(shape = (1, seq_transformer_info['Transformer_fea_len']))

    if not hp:
        x = keras.layers.GlobalMaxPool1D()(model_in)
        # print('-'*30)
        # print(x)
        # print('-'*30)
    else:    
        convs=[]
        bns = []
        
        strides = [1]*len(hp['filters'])
        if hp['strides']:
            strides=hp['strides']
        
        for f, k, d, s in zip(hp['filters'],hp['kernel_widths'],hp['dilations'],strides):
            convs.append(keras.layers.Conv1D(f,
                                    kernel_size=k,
                                    dilation_rate=d,
                                    strides=s,
                                    kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=hp['L2_conv']),
                                    padding='same')
                        )
            bns.append(keras.layers.BatchNormalization())                    
        
        elu_activation = keras.layers.ELU()
        
        dropout_conv = keras.layers.Dropout(hp['dropout_conv'])
        
        x = keras.layers.BatchNormalization()(model_in)
        
        for c,bn in zip(convs,bns):
            x = c(x)
            x = elu_activation(x)
            x = dropout_conv(x)
            x = bn(x)
            
        x = keras.layers.GlobalMaxPool1D()(x)
    
    # print('-'*30)
    # print(x)
    # print('-'*30)
    return keras.Model(inputs=model_in, outputs=x, name=name)

def conv_structure_extractor(hp,stru_len,fea_len,name=None):
    """ convolutional network for sequences
    
    The sequence input will initially be embedded by a trainable 
    embedding layer.
    
    Multiple 1D convolutions along a sequence can be applied, followed
    by a global max pool operation at the last convolutional output.
    
    Convolutions can be standard or dilated.
    
    parameters
    -----------
    
    hp: dict
        A dictionary of hyperparameters for the model:
         - embed_dim: int
                 dimension to embed each element of the sequences
         - filters: list
                 list of ints, element i of the list is the number of 
                 filters to put in the i'th conv layer
        - strides: list
                 list of ints, element i of the list is the strides to 
                 use in the i'th conv layer. strides[i] must be 1 where
                 dilations[i]=/=1
        - kernel_widths: list
                 list of ints, element i of the list is the width of the  
                 kernel to use in the filter of the i'th conv layer
        - dilations: list
                list of ints for the dilation of conv layer i.
        - L2_conv: float
                value for L2 norm penalty for each conv layer
        - dropout_conv: float
                dropout to apply following each conv layer
                
    returns
    --------
    model: tf.keras.Model
            Keras model that converts input into a a feature representation. i.e prior
            to any final dense layers.
    
    
    """
    model_in = keras.Input(shape = (stru_len,fea_len))

    convs=[]
    bns = []
    
    strides = [1]*len(hp['filters'])
    if hp['strides']:
        strides=hp['strides']
    
    for f,k,d,s in zip(hp['filters'],hp['kernel_widths'],hp['dilations'],strides):
        convs.append( keras.layers.Conv1D(f,
                                  kernel_size=k,
                                  dilation_rate=d,
                                  strides=s,
                                  kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=hp['L2_conv']),
                                  padding='same')
                    )
        bns.append(keras.layers.BatchNormalization())                    
    
    elu_activation = keras.layers.ELU()
    
    dropout_conv = keras.layers.Dropout(hp['dropout_conv'])
    
    x = keras.layers.BatchNormalization()(model_in)
    
    for c,bn in zip(convs,bns):
        x = c(x)
        x = elu_activation(x)
        x = dropout_conv(x)
        x = bn(x)
        
    x = keras.layers.GlobalMaxPool1D()(x)
    
    return keras.Model(inputs=model_in, outputs=x, name=name)

def vj_extractor(hp, name):
    """ Extract Gene info - embed and dropout
    
    parameters
    -----------
    hp: dict
        dictionary of hyperpareameters, keys:
         - vj_embed : int
             the dimension in which to embed the one-hot gene representation
         - vj_width: int
             the original dimension of the one-hot representation coming in
         - dropout: float
             how much dropout to apply after the embedding
             
    name: string
        name to give the extractor model
        
    returns
    --------
    model: tf.keras.Model
            Keras model that converts input into a a feature representation. i.e prior
            to any final dense layers.
    
    """
    vj_in = keras.Input(shape = (1,), name='vj_input')
    
    reshape_layer = keras.layers.Reshape([hp['vj_embed']])
    
    # hp['vj_width'] = possible vector length (vocabulary)
    x_vj_mu = keras.layers.Embedding(hp['vj_width'], hp['vj_embed'])(vj_in)
    
    x_vj_mu = reshape_layer(x_vj_mu)
    
    x_vj_mu = keras.layers.Dropout(hp['dropout'])(x_vj_mu)

    return keras.Model(inputs=vj_in, outputs=x_vj_mu, name=name)

def conv_seq_extractor_no_embed(hp,seq_len,feature_dim,name=None):
    """ convolutional network for sequences
    
    The sequence input will initially be embedded by a trainable 
    embedding layer.
    
    Multiple 1D convolutions along a sequence can be applied, followed
    by a global max pool operation at the last convolutional output.
    
    Convolutions can be standard or dilated.
    
    parameters
    -----------
    
    hp: dict
        A dictionary of hyperparameters for the model:
         - embed_dim: int
                 dimension to embed each element of the sequences
         - filters: list
                 list of ints, element i of the list is the number of 
                 filters to put in the i'th conv layer
        - strides: list
                 list of ints, element i of the list is the strides to 
                 use in the i'th conv layer. strides[i] must be 1 where
                 dilations[i]=/=1
        - kernel_widths: list
                 list of ints, element i of the list is the width of the  
                 kernel to use in the filter of the i'th conv layer
        - dilations: list
                list of ints for the dilation of conv layer i.
        - L2_conv: float
                value for L2 norm penalty for each conv layer
        - dropout_conv: float
                dropout to apply following each conv layer
                
    returns
    --------
    model: tf.keras.Model
            Keras model that converts input into a a feature representation. i.e prior
            to any final dense layers.
    
    
    """
    model_in = keras.Input(shape = (seq_len,feature_dim))
    
    convs=[]
    bns = []
    
    strides = [1]*len(hp['filters'])
    if hp['strides']:
        strides=hp['strides']
    
    for f,k,d,s in zip(hp['filters'],hp['kernel_widths'],hp['dilations'],strides):
        convs.append( keras.layers.Conv1D(f,
                                  kernel_size=k,
                                  dilation_rate=d,
                                  strides=s,
                                  kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.0, l2=hp['L2_conv']),
                                  padding='same')
                    )
        bns.append( keras.layers.BatchNormalization() )
                    
    
    elu_activation = keras.layers.ELU()
    
    dropout_conv = keras.layers.Dropout(hp['dropout_conv'])
    
    x=model_in
    for c,bn in zip(convs,bns):
        x = c(x)
        x = elu_activation(x)
        x = dropout_conv(x)
        x = bn(x)
        
    x = keras.layers.GlobalMaxPool1D()(x)
    
    return keras.Model(inputs=model_in, outputs=x, name=name)