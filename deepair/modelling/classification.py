import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow.keras as keras
from abc import ABC
from deepair.modelling.base_models import BaseXCRModelWithProcessors, BaseXCRModel

#%%
class SeqClassificationModelWithProcessor(BaseXCRModelWithProcessors):
    """ model for sequences wrapped with processor to convert text to numeric input
    """
    
    def run(self,inputs):
        """ run the model on given input
        
        Model will be run in inference mode
        
        parameters
        -----------
        inputs: dict
            keyed dictionary of inputs. Keys may be e.g., 'Vgene','TRB_cdr3' etc.
            Values should be lists or arrays of input data. Strings such a cdr3 regions 
            can then be automatically processed by the internal processors of the same key.
            See BaseXCRModelWithProcessors.
        
        returns
        -------
        x: tf.Tensor
            Predictions of the model on each input sample.
        """
        return self.simple_predict(inputs)
    
    def fit(self, inputs, labels, **kwargs):
        """ Fit the model on given input
        
        parameters
        -----------
        inputs: dict
            keyed dictionary of inputs. Keys may be e.g., 'Vgene','TRB_cdr3' etc.
            Values should be lists or arrays of input data. Strings such a cdr3 regions 
            can then be automatically processed by the internal processors of the same key.
            See BaseXCRModelWithProcessors.
            
        labels: list-like
            list/array of labels for the inputs. Should have length the same as the values 
            in the input dictionary.
            
        kwargs: optional keyword arguments
            Kwargs should be those accepted by keras.Model.fit()
        
        returns
        --------
        history: tf.keras.History
            A tf.keras history object storing the fitting process results per epoch.
            
        """
        fit_input = self.process_pred_input(inputs)
        if 'validation_data' in kwargs:
            vali_in_dict = self.process_pred_input(kwargs['validation_data'][0])
            v_labels = kwargs['validation_data'][1]
            if len(kwargs['validation_data'])==3:
                v_sw = kwargs['validation_data'][2]
                vali_tuple = (vali_in_dict,v_labels,v_sw)
            else:
                vali_tuple = (vali_in_dict,v_labels)
            kwargs.update({'validation_data': vali_tuple})
            
        return self.model.fit(fit_input,labels,**kwargs)
    
    def evaluate(self, inputs, labels, **kwargs):
        """ Evaluate model performance on given input
        
        Note that evaluating functions can be added to the [...] ModelWithProcessors object
        via the add_evaluator() method - seee BaseXCRModelWithProcessors class in modelling/base_models.
        
        parameters
        -----------
        inputs: dict
            keyed dictionary of inputs. Keys may be e.g., 'Vgene','TRB_cdr3' etc.
            Values should be lists or arrays of input data. Strings such a cdr3 regions 
            can then be automatically processed by the internal processors of the same key.
            See BaseXCRModelWithProcessors.
            
        labels: list-like
            list/array of labels for the inputs. Should have length the same as the values 
            in the input dictionary.
            
        kwargs: optional keyword arguments
            Kwargs should be those accepted by keras.Model.fit()
        
        returns
        --------
        score_dict: dict
            A dictionary of evaluations keyed by the names that were provided when
            evaluator functions were added to the [...]ModelWithProcessors object.
        """
        preds = self.run(inputs)
        if tf.is_tensor(preds):
            preds = preds.numpy()
        score_dict = self._apply_evaluations(preds,labels)
        return score_dict
    
    def explain(self, inputs, output_channel, **kwargs):
        """ 
        Generate the explaination of the model based on the Integrated Gradients method
        
        parameters
        -----------
        inputs: dict
            keyed dictionary of inputs. Keys may be e.g., 'Vgene','TRB_cdr3' etc.
            Values should be lists or arrays of input data. Strings such a cdr3 regions 
            can then be automatically processed by the internal processors of the same key.
            See BaseXCRModelWithProcessors.
            
        labels: list-like
            list/array of labels for the inputs. Should have length the same as the values 
            in the input dictionary.
            
        kwargs: optional keyword arguments
            Kwargs should be those accepted by keras.Model.fit()
        
        returns
        --------
        score_dict: dict
            The ex
        """

        pred_input = self.process_pred_input(inputs)
        explain = self.model.explain(inputs = pred_input, output_channel = output_channel)
        return explain

class FusionModel(BaseXCRModel):
    """ Model for classifying X-cell-receptors with V/J and sequence info"""
    
    def call(self,inputs,training=None):
        """ call the model
        parameters
        -----------

        inputs: dict
            A dictionary with keys as the names of the encoders 
            provided on initialization. (see BaseXCR class for further info).
            Values should be arrays of the input data for that keyed input. 
            E.g you may have 'Vgene' as a key for one your inputs, and the values
            might be [3,5,7,2,9,7].

        training: bool, optional, default=None
            If False, treat as inference, if True, trat as training.
            Keras model fit()/evaluate()/test() will appropriately select. 

        returns
        -------
        x: tf.Tensor
            The model's prediction for all samples provided in inputs
        """
    
        z = self._calculate_z(inputs,training=training)
        x = self.predictor(z,training=training)
        return x

class GatedFusionModel(keras.Model, ABC):
    """ Base model for XCR problems
    parameters
    ----------
    
    seq_encoders: (dict)
            dictionary of sequence encoders : {'encoder name' : keras.Model }. The model will expect 
            the input data to be a dictionary with keys of sequences matching the keys of the required
            encoder.
            
    extra encoders: (dict)
            dictionary of encoders for extra variables: {'encoder name' : keras.Model }. 
            The model will expect the input data to be a dictionary with keys of extra variables 
            matching the keys of the required encoder. 
            
    predictor: (tf.keras.Model)
            A keras model that will take the features of all the encoded inputs, and return a prediction.
            
    input_list: (list, optional, default=None)
            A list of keys of the inputs that the model should actually process. If None, input list
            will default to an alphabetically sorted list of all keys in the seq_encoders and gene_encoders
            
    **kwargs: optional keyword arguments to be passed to the keras.Model superclass.    
    
    """
    
    def __init__(self,
                 seq_encoders,
                 str_encoders,
                 gene_encoders,
                 gated_fuison,
                 input_list=None,
                 **kwargs
                ):
        super(GatedFusionModel, self).__init__(**kwargs)
        self.seq_encoders = seq_encoders
        self.str_encoders = str_encoders
        self.gene_encoders = gene_encoders
        if input_list is None:
            input_list = sorted(list(seq_encoders.keys()))+sorted(list(gene_encoders.keys()))+sorted(list(str_encoders.keys()))
        self.input_list = input_list        
        self.z_concat = keras.layers.Concatenate(name='z_concat')
        self.predictor = FusionrNet(gated_fuison)
            
    def call(self, inputs, training=None):
        """ call the model
        
        parameters
        -----------

        inputs: dict
            A dictionary with keys as the names of the encoders 
            provided on initialization. (see BaseXCR class for further info).
            Values should be arrays of the input data for that keyed input. 
            E.g you may have 'Vgene' as a key for one your inputs, and the values
            might be [3,5,7,2,9,7].

        training: bool, optional, default=None
            If False, treat as inference, if True, trat as training.
            Keras model fit()/evaluate()/test() will appropriately select. 

        returns
        -------
        x: tf.Tensor
            The model's prediction for all samples provided in inputs
        """
        input_list_seq = [item for item in self.input_list if not('_Stru' in item)]
        input_list_stru = [item for item in self.input_list if '_Stru' in item]
        z_seq = self._calculate_z(inputs, selection=input_list_seq, training=training)
        z_stru = self._calculate_z(inputs, selection=input_list_stru, training=training)
        x = self.predictor([z_seq,z_stru], training=training)
        return x
        
        
    def _calculate_z(self, inputs, selection=None, training=None):
        """ calculate latent space features
        
        parameters
        ----------
        
        inputs: (dict)
                dict of (np/tf) arrays of the inputs, keys should be the names
                of the inputs, and values the arrays.
                
        selection: (list, optional, default=None)
                only those inputs whose keys are in selection will be processed into
                the feature space. If None, the self.input_list will be used
                
        returns
        -------
        
        z: (np or tf array)
            The feature space vector of the inputs appearing in selection_list.
        
        """
        if selection is None:
            selection=self.input_list
        
        vals = []
        for sel in selection:
            if sel in list(self.seq_encoders.keys()):
                try:
                    x_e = self.seq_encoders[sel](inputs[sel],training=training)
                except:
                    raise RuntimeError('check sequence encoder')
                vals.append(x_e)
            elif sel in list(self.str_encoders.keys()):
                try:
                    x_s = self.str_encoders[sel](inputs[sel],training=training)
                except:
                    raise RuntimeError('check structure encoder')
                vals.append(x_s)
            elif sel in list(self.gene_encoders.keys()):
                vals.append(self.gene_encoders[sel](inputs[sel],training=training))
            else:
                raise RuntimeError(f"unknown selection key passed: {sel}")
        
        if len(vals)>1:
            z = self.z_concat(vals)
        else:
            z = vals[0]
        return z   

class FusionrNet(tf.keras.Model, ABC):

    def __init__(self, options):
        super(FusionrNet, self).__init__()
        self.fusion = BilinearFusion(skip=options['skip'], 
                                        gate1=options['seq_gate'], 
                                        gate2=options['stru_gate'], 
                                        gated_fusion=options['gated_fusion'],
                                        dim1=options['seq_dim'], 
                                        dim2=options['stru_dim'], 
                                        scale_dim1=options['seq_scale'], 
                                        scale_dim2=options['stru_scale'], 
                                        mmhid1=options['mmhid1'], 
                                        mmhid2=options['mmhid2'], 
                                        dropout_rate=options['dropout_rate'])
        self.classifier = tf.keras.layers.Dense(options['label_dim'], activation=options['activation'])


    def call(self, seq_stru_features, training=False):        
        features = self.fusion(seq_stru_features)
        Y_prob = self.classifier(features)
        return Y_prob

class BilinearFusion(tf.keras.Model, ABC):
    # we do not offer a bilinear transformation to the incoming data y = x1Ax2+b
    def __init__(self, skip=True,
                 gate1=True, gate2=True, gated_fusion=True,
                 dim1=32, dim2=32, 
                 scale_dim1=1, scale_dim2=1, 
                 mmhid1=64, mmhid2=16, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.gate1 = gate1
        self.gate2 = gate2
        self.gated_fusion = gated_fusion

        # dim1_og, dim2_og = dim1, dim2 
        dim1, dim2 = int(dim1/scale_dim1), int(dim2/scale_dim2)

        # skip_dim = dim1+dim2+2 if skip else 0
        
        # A sequential container. Modules will be added to it in the order they are passed in the constructor. 
        # Alternatively, an ordered dict of modules can also be passed in.
        self.linear_h1 = tf.keras.layers.Dense(dim1,activation='relu')
        # we do not offer a bilinear transformation to the incoming data y = x1Ax2+b
        self.linear_z1 = tf.keras.layers.Dense(dim1)
        if self.gate1:
            self.linear_o1 = tf.keras.Sequential()
            self.linear_o1.add(tf.keras.layers.Dense(dim1,activation='relu'))
            self.linear_o1.add(tf.keras.layers.Dropout(rate = dropout_rate))

        self.linear_h2 = tf.keras.layers.Dense(dim2,activation='relu')
        self.linear_z2 = tf.keras.layers.Dense(dim2)
        if self.gate2:
            self.linear_o2 = tf.keras.Sequential()
            self.linear_o2.add(tf.keras.layers.Dense(dim2,activation='relu'))
            self.linear_o2.add(tf.keras.layers.Dropout(rate = dropout_rate))

        self.post_fusion_dropout = tf.keras.layers.Dropout(rate = dropout_rate)
        self.encoder1 = tf.keras.Sequential()
        self.encoder1.add(tf.keras.layers.Dense(mmhid1,activation='relu'))
        self.encoder1.add(tf.keras.layers.Dropout(rate = dropout_rate))
        self.encoder2 = tf.keras.Sequential()
        self.encoder2.add(tf.keras.layers.Dense(mmhid1,activation='relu'))
        self.encoder2.add(tf.keras.layers.Dense(mmhid2,activation='relu'))
        self.encoder2.add(tf.keras.layers.Dropout(rate = dropout_rate))

        self.concat_z1 = keras.layers.Concatenate()
        self.concat_z2 = keras.layers.Concatenate()
        self.concat_3 = keras.layers.Concatenate()
        self.concat_4 = keras.layers.Concatenate()
        self.flatten = tf.keras.layers.Flatten(data_format='channels_first')

    def call(self, vec1_vec2, training=False):
    ### Gated Multimodal Units
        vec1 = vec1_vec2[0]
        vec2 = vec1_vec2[1]
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(self.concat_z1([vec1, vec2]))
            o1 = self.linear_o1(tf.keras.activations.sigmoid(z1)*h1)
        else:
            # print(vec1)
            o1 = self.linear_h1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(self.concat_z2([vec1, vec2]))
            o2 = self.linear_o2(tf.keras.activations.sigmoid(z2)*h2)
        else:
            # print(vec2)
            o2 = self.linear_h2(vec2)

        ### Fusion
        if self.gated_fusion:
            o1_m = tf.expand_dims(o1,2)
            o2_m = tf.expand_dims(o2,1)

            o12 = self.flatten(tf.matmul(o1_m, o2_m)) # BATCH_SIZE X 1024
            out = self.post_fusion_dropout(o12)
            out = self.encoder1(out)
            if self.skip:                 
                out = self.concat_3([out, o1, o2])
        else:
            out = self.concat_4([o1, o2])
        out = self.encoder2(out)
        return out

#%%
class GatedFusionModelMultiTask(keras.Model, ABC):
    """ Base model for XCR problems
    parameters
    ----------
    
    seq_encoders: (dict)
            dictionary of sequence encoders : {'encoder name' : keras.Model }. The model will expect 
            the input data to be a dictionary with keys of sequences matching the keys of the required
            encoder.

    str_encoders: (dict)
            
    gene_encoders: (dict)
            dictionary of encoders for gene variables: {'encoder name' : keras.Model }. 
            The model will expect the input data to be a dictionary with keys of extra variables 
            matching the keys of the required encoder. 
            
    predictor: (tf.keras.Model)
            A keras model that will take the features of all the encoded inputs, and return a prediction.
            
    input_list: (list, optional, default=None)
            A list of keys of the inputs that the model should actually process. If None, input list
            will default to an alphabetically sorted list of all keys in the seq_encoders and gene_encoders
            
    **kwargs: optional keyword arguments to be passed to the keras.Model superclass.    
    
    """
    
    def __init__(self,
                 seq_encoders,
                 str_encoders,
                 gene_encoders,
                 gated_fuison,
                 input_list=None,
                 **kwargs
                ):
        super(GatedFusionModelMultiTask, self).__init__(**kwargs)
        self.seq_encoders = seq_encoders
        self.str_encoders = str_encoders
        self.gene_encoders = gene_encoders
        if input_list is None:
            input_list = sorted(list(seq_encoders.keys()))+sorted(list(gene_encoders.keys()))+sorted(list(str_encoders.keys()))
        self.input_list = input_list        
        self.z_concat = keras.layers.Concatenate(name='z_concat')
        self.predictor = FusionrNetMultiTask(gated_fuison)
        
    def call(self,inputs,training=None):
        """ call the model
        parameters
        -----------

        inputs: dict
            A dictionary with keys as the names of the encoders 
            provided on initialization. (see BaseXCR class for further info).
            Values should be arrays of the input data for that keyed input. 
            E.g you may have 'Vgene' as a key for one your inputs, and the values
            might be [3,5,7,2,9,7].

        training: bool, optional, default=None
            If False, treat as inference, if True, trat as training.
            Keras model fit()/evaluate()/test() will appropriately select. 

        returns
        -------
        x: tf.Tensor
            The model's prediction for all samples provided in inputs
        """
        input_list_seq = [item for item in self.input_list if not('_Stru' in item)]
        input_list_stru = [item for item in self.input_list if '_Stru' in item]
        z_seq = self._calculate_z(inputs, selection=input_list_seq, training=training)
        z_stru = self._calculate_z(inputs, selection=input_list_stru, training=training)
        Y_reg, Y_prob = self.predictor([z_seq,z_stru], training=training)
        return [Y_reg, Y_prob]

        
    def _calculate_z(self, inputs, selection=None, training=None):
        """ calculate latent space features
        
        parameters
        ----------
        
        inputs: (dict)
                dict of (np/tf) arrays of the inputs, keys should be the names
                of the inputs, and values the arrays.
                
        selection: (list, optional, default=None)
                only those inputs whose keys are in selection will be processed into
                the feature space. If None, the self.input_list will be used
                
        returns
        -------
        
        z: (np or tf array)
            The feature space vector of the inputs appearing in selection_list.
        
        """
        if selection is None:
            selection=self.input_list
        
        vals = []
        for sel in selection:
            if sel in list(self.seq_encoders.keys()):
                try:
                    x_e = self.seq_encoders[sel](inputs[sel],training=training)
                except:
                    raise RuntimeError('check sequence encoder')
                vals.append(x_e)
            elif sel in list(self.str_encoders.keys()):
                try:
                    x_s = self.str_encoders[sel](inputs[sel],training=training)
                except:
                    raise RuntimeError('check structure encoder')
                vals.append(x_s)
            elif sel in list(self.gene_encoders.keys()):
                vals.append(self.gene_encoders[sel](inputs[sel],training=training))
            else:
                raise RuntimeError(f"unknown selection key passed: {sel}")
        
        if len(vals)>1:
            z = self.z_concat(vals)
        else:
            z = vals[0]
        return z   

class FusionrNetMultiTask(tf.keras.Model, ABC):

    def __init__(self, options):
        super(FusionrNetMultiTask, self).__init__()
        self.fusion = BilinearFusionMultiTask(skip=options['skip'], 
                                            gate1=options['seq_gate'], 
                                            gate2=options['stru_gate'], 
                                            gated_fusion=options['gated_fusion'],
                                            dim1=options['seq_dim'], 
                                            dim2=options['stru_dim'], 
                                            scale_dim1=options['seq_scale'], 
                                            scale_dim2=options['stru_scale'], 
                                            mmhid=options['mmhid'], 
                                            dropout_rate=options['dropout_rate'])        
        # regression 
        self.output1 = tf.keras.Sequential()
        self.output1.add(tf.keras.layers.Dense(16,activation='relu'))
        self.output1.add(tf.keras.layers.Dense(options['label_dim'], activation=options['activation'], name = 'out1'))     

        self.output2 = tf.keras.Sequential()
        self.output2.add(tf.keras.layers.Dense(16,activation='relu'))
        self.output2.add(tf.keras.layers.Dense(8,activation='relu'))
        self.output2.add(tf.keras.layers.Dense(1, activation='softmax', name = 'out2'))


    def call(self, seq_stru_features, training=False):        
        features = self.fusion(seq_stru_features)
        Y_reg = self.output1(features)
        Y_prob = self.output2(features)
        return Y_reg, Y_prob

class BilinearFusionMultiTask(tf.keras.Model, ABC):
    # we do not offer a bilinear transformation to the incoming data y = x1Ax2+b
    def __init__(self, skip=True,
                 gate1=True, gate2=True, gated_fusion=True,
                 dim1=32, dim2=32, 
                 scale_dim1=1, scale_dim2=1, 
                 mmhid=64, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.gate1 = gate1
        self.gate2 = gate2
        self.gated_fusion = gated_fusion

        # dim1_og, dim2_og = dim1, dim2 
        dim1, dim2 = int(dim1/scale_dim1), int(dim2/scale_dim2)

        # skip_dim = dim1+dim2+2 if skip else 0
        
        # A sequential container. Modules will be added to it in the order they are passed in the constructor. 
        # Alternatively, an ordered dict of modules can also be passed in.
        self.linear_h1 = tf.keras.layers.Dense(dim1,activation='relu')
        # we do not offer a bilinear transformation to the incoming data y = x1Ax2+b
        self.linear_z1 = tf.keras.layers.Dense(dim1)
        if self.gate1:
            self.linear_o1 = tf.keras.Sequential()
            self.linear_o1.add(tf.keras.layers.Dense(dim1,activation='relu'))
            self.linear_o1.add(tf.keras.layers.Dropout(rate = dropout_rate))

        self.linear_h2 = tf.keras.layers.Dense(dim2,activation='relu')
        self.linear_z2 = tf.keras.layers.Dense(dim2)
        if self.gate2:
            self.linear_o2 = tf.keras.Sequential()
            self.linear_o2.add(tf.keras.layers.Dense(dim2,activation='relu'))
            self.linear_o2.add(tf.keras.layers.Dropout(rate = dropout_rate))

        self.post_fusion_dropout = tf.keras.layers.Dropout(rate = dropout_rate)
        self.encoder1 = tf.keras.Sequential()
        self.encoder1.add(tf.keras.layers.Dense(mmhid,activation='relu'))
        self.encoder1.add(tf.keras.layers.Dropout(rate = dropout_rate))
        self.encoder2 = tf.keras.Sequential()
        self.encoder2.add(tf.keras.layers.Dense(mmhid,activation='relu'))
        self.encoder2.add(tf.keras.layers.Dropout(rate = dropout_rate))

        self.concat_z1 = keras.layers.Concatenate()
        self.concat_z2 = keras.layers.Concatenate()
        self.concat_3 = keras.layers.Concatenate()
        self.concat_4 = keras.layers.Concatenate()
        self.flatten = tf.keras.layers.Flatten(data_format='channels_first')

    def call(self, vec1_vec2, training=False):
    ### Gated Multimodal Units
        vec1 = vec1_vec2[0]
        vec2 = vec1_vec2[1]
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(self.concat_z1([vec1, vec2]))
            o1 = self.linear_o1(tf.keras.activations.sigmoid(z1)*h1)
        else:
            # print(vec1)
            o1 = self.linear_h1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(self.concat_z2([vec1, vec2]))
            o2 = self.linear_o2(tf.keras.activations.sigmoid(z2)*h2)
        else:
            # print(vec2)
            o2 = self.linear_h2(vec2)

        ### Fusion
        if self.gated_fusion:
            o1_m = tf.expand_dims(o1,2)
            o2_m = tf.expand_dims(o2,1)

            o12 = self.flatten(tf.matmul(o1_m, o2_m)) # BATCH_SIZE X 1024
            out = self.post_fusion_dropout(o12)
            out = self.encoder1(out)
            if self.skip:                 
                out = self.concat_3([out, o1, o2])
        else:
            out = self.concat_4([o1, o2])
        out = self.encoder2(out)
        return out