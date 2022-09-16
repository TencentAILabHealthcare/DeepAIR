import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow.keras as keras
from abc import ABC, abstractmethod
import pickle
import os
import numpy as np

#%%
def one_hot_decode(encoded_data, axis=-1):
    return np.argmax(encoded_data, axis=axis)

class RegressionModelWithProcessor(tf.keras.Model):
    """ 
    model for sequences wrapped with processor to convert text to numeric input
    """
    def __init__(self, model, processors={}, stru_processors={}, extra_tokenizers={}, load=False):
        super(RegressionModelWithProcessor, self).__init__()
        self.model = model
        self.processors = processors
        self.stru_processors = stru_processors        
        self.tokenizers = extra_tokenizers
        self.evaluators = dict()
        

    def process_pred_input(self, inputs):
        """ process inputs into format to be passed to keras model
        
        parameters
        ----------
        inputs: (dict) 
                dictionary of named inputs, {name: input data}. input data typically
                an array of strings.                
        returns
        ----------
        in_dict: (dict)
                dictionary of processed outputs for the inputs that correspond 
                to keys of self.processors and self.tokenizers.        
        """
        pro_dict = self.process_seqs(inputs) # sequence
        tok_dict = self.tokenize(inputs) # vdj gene
        return {**inputs, **pro_dict, **tok_dict}

    def process_seqs(self, inputs):
        """ process input sequences      

        process the inputs that correspond to named processors in self.processors.
        These are inputs that are CDRs.        
        
        parameters
        -----------        
        inputs: (dict) 
                dictionary of named inputs, {name: input data}. input data typically
                an array of strings.                
        returns
        ----------
        pro_dict: (dict)
                dictionary of processed outputs for the inputs that correspond 
                to a processor key of self.processors.                
        """
        pro_dict = dict()
        for key,value in self.processors.items():
            #pro_dict[key] = value.transform_seqs(inputs[key])
            if value:
                pro_dict[key] = value.transform(inputs[key])['seqs']
        return pro_dict

    def tokenize(self,inputs):
        """ Apply tokenizers to inputs
        
        parameters
        -----------        
        inputs: (dict) 
                dictionary of named inputs, {name: input data}. input data typically
                an array of strings.                
        returns
        ----------
        tokenized: (dict)
                dictionary of tokenized outputs for the inputs that correspond 
                to a tokenizer key of self.tokenizers.         
        """
        tokenized = dict()
        for tok_name, tok in self.tokenizers.items():
            tokenized[tok_name] = np.array(tok.texts_to_sequences(inputs[tok_name]))
        return tokenized
    
    def unprocess(self,predictions):
        """ Unprocess predicted sequences
        
        From a sequence of integers representing a sequence as it would be 
        encoded by a processor of the model, revert the sequence into it's
        unprocessed (amino acid) form.
        
        predictions will be processed in place into sequences of chars.
        
        parameters
        -----------
        predictions: (dict)
                A dictionary of predictions of (integer) sequences. keys of predictions
                should match the keys of self.processors. values should be arrays of 
                lists of integers.
                
        returns
        ---------
        None
        
        """
        pred_seqs = dict()
        for k, v in self.processors.items():
            argmaxed = np.argmax(predictions[k], axis=-1).astype(int)
            predictions[k] = v.seqs_to_text(argmaxed)
        return None
    
    def calculate_z(self, inputs, selection=None, first_seq=None):
        """ calculate the latent space / feature space ,z, of the model
        
        parameters
        -----------
        
        inputs: (dict)
                input dictionary from which to calculate_z
                
        selection: (list, optional, default=None)
                only those input names appearing in selection will be processed into
                z. If None, all inputs will be processed that appear in self.model's
                input list.
                
        first_seq: (string, optional, default=None):
                the sequence input with this key will appear as the first vector in z.
                If the key also appears in selection, the encoding will appear twice in
                z.
        
        returns
        ---------
        z: (np.array)
            numpy array of z. dimension (n_samples, (z_dim))      
        
        """
        
        if selection is None:
            #selection=sorted(list(self.processors.keys())) + sorted(list(self.tokenizers.keys()))
            selection=self.model.input_list
        
        if not all([((sel in self.processors) or (sel in self.stru_processors) or (sel in self.tokenizers)) for sel in selection]):
            raise ValueError('Unknown key provided')
            
        in_dict = self.process_pred_input(inputs)
        
        z=[]
        
        if first_seq is not None:
            this_enc = self.model.seq_encoders[first_seq](in_dict[first_seq])
            if isinstance(this_enc, dict):
                x_e = this_enc['mu'].numpy()
                z.append(x_e)
            #selection = set(self.processors.keys())-set(first_seq)                    
        
        for sel in selection:
            if sel in list(self.processors.keys()):
                this_enc = self.model.seq_encoders[sel](in_dict[sel])
                if isinstance(this_enc, dict):
                    x_e = this_enc['mu'].numpy()
                else:
                    x_e = this_enc.numpy()
                z.append(x_e)
            elif sel in list(self.stru_processors.keys()):
                this_enc = self.model.str_encoders[sel](in_dict[sel])
                x_e = this_enc.numpy()
                z.append(x_e)
            elif sel in list(self.tokenizers.keys()):
                z.append(self.model.extra_encoders[sel](in_dict[sel]).numpy())
            else:
                try:
                    z.append(self.model.extra_encoders[sel](in_dict[sel]).numpy())
                except:
                    raise RuntimeError("TCRAI ERROR: unknown input key passed to Model")    
        return np.hstack(z)  
    
    def simple_predict(self,inputs):
        """ make predictions from inputs
        
        inputs: (dict) 
                dictionary of named inputs, {name: input data}. input data typically
                an array of strings.
                
        returns
        ----------
        in_dict: (dict)
                dictionary of processed outputs for the inputs that correspond 
                to keys of self.processors and self.tokenizers.                       
        """
        pred_input = self.process_pred_input(inputs)
        # preds = self.model.predict(pred_input)
        key_list = list(pred_input.keys())
        for key in key_list:
            if  key in ['TRB_cdr3_Stru', 'TRA_cdr3_Stru', 'TRB_cdr3_splited', 'TRA_cdr3_splited']:
                pred_input[key] = pred_input[key].astype('float32')
            elif key in ['ID','TRB_cdr3','TRA_cdr3']:
                pred_input[key] = tf.expand_dims(pred_input[key], axis=1)
            elif key in ['A1101_AVFDRKSDAK_EBNA-3B_EBV_Reg_Stage',  
                        'A0201_GILGFVFTL_Flu-MP_Influenza_Reg_Stage',
                        'A1101_IVTDFSVIK_EBNA-3B_EBV_Reg_Stage',
                        'B0801_RAKFKQLL_BZLF1_EBV_Reg_Stage',
                        'A0201_GLCTLVAML_BMLF1_EBV_Reg_Stage',
                        'A0201_ELAGIGILTV_MART-1_Cancer_Reg_Stage',
                        'A0301_KLGGALQAK_IE-1_CMV_Reg_Stage']:
                pred_input[key] = tf.expand_dims(pred_input[key], axis=1)
        preds = self.model.call(pred_input)
        return preds
    
    def compile(self, **kwargs):
        """ compile the internal keras model
        
        parameters
        ------------
        
        **kwargs: keyword arguments as expected by keras model compile method.
        
        returns
        --------
        None        
        """
        self.model.compile(**kwargs)
    
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
    

    def fit(self,inputs,labels,**kwargs):
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
            
        return self.model.fit(fit_input, labels, **kwargs)
    
    def add_evaluator(self, evaluator, name, position):
        """ add an evaluator with a name for later retrieval
        
        call evaluate() to see the results of the evaluation
        
        parameters
        ------------
        evaluator: (Callable) 
                callable taking inputs (y_true, y_pred)
        name: (string)
                A name for the evaluator
                    
        """
        self.evaluators[name] = [evaluator, position]

    def remove_evaluator(self,name):
        """ remove an evaluator
        
        parameters
        ------------
        name : (string)
                The name of the evaluator to be removed
        
        """
        del self.evaluators[name]

    def evaluate(self, inputs, labels, **kwargs):
        """ Evaluate model performance on given input
        
        Note that evaluating functions can be added to the [...]ModelWithProcessors object
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
        score_dict = self._apply_evaluations(preds, labels)
        return score_dict
   
    def _apply_evaluations(self, predictions, labels):
        """ Calculate evaluation metrics as per internal evaluators
        
        parameters
        -----------
        predictions: (np.array)
                output predictions from the model
                
        labels: (np.array)
                Array of the labels
                
        returns 
        --------
        score_dict: (dict)
                Dictionary with keys as the names of the internal evaluators, and 
                values as the output of those evaluators
        
        """
        # print('-'*30)
        # print(predictions)
        # print(labels)
        score_dict = dict()
        # decode one hot
        predictions[1] = one_hot_decode(predictions[1], axis=-1)

        for name, evaluation in self.evaluators.items():
            evaluation_fun = evaluation[0]
            evaluation_pos = evaluation[1]
            if name == 'ACC':
                score_dict[name] = evaluation_fun(labels[evaluation_pos], predictions[evaluation_pos]>0.5)
            else:
                score_dict[name] = evaluation_fun(labels[evaluation_pos], predictions[evaluation_pos])
        
        return score_dict
    
    def _save_model(self,model_path):
        self.model.save(model_path)
        
    def save(self,directory):
        """ Save the model, and its processors and tokenizers
        
        The keras model, processors, tokenizers, and the internal model's input list
        will all be save into the given directory. Allowing for easy later loading of the 
        keras model, and the processors etc, so one doesn't have to worry about strong all
        the processors separtely.
        
        parameters
        ----------
        directory: (string, path)
                A string for the path to a directory in whcih to store the model. if the directory 
                doesn't exist, it will be created.
                
        returns 
        --------        
        None               
        """
        model_path = os.path.join(directory, 'model')
        #self.model.save(model_path)
        self._save_model(model_path)

        # pickle processor, save model
        pickle_path = os.path.join(directory, 'processors')
        if not os.path.exists(pickle_path):
            os.makedirs(pickle_path)
        pickle.dump(self.processors, open(os.path.join(pickle_path,'processors.pickle'), "wb"))
        
        # pickle processor, save model
        pickle_path = os.path.join(directory, 'stru_processors')
        if not os.path.exists(pickle_path):
            os.makedirs(pickle_path)
        pickle.dump(self.stru_processors, open(os.path.join(pickle_path,'stru_processors.pickle'), "wb"))
        
        pickle_path = os.path.join(directory, 'extra_tokenizers')
        if not os.path.exists(pickle_path):
            os.makedirs(pickle_path)
        pickle.dump( self.tokenizers, open(os.path.join(pickle_path,'tokenizers.pickle'), "wb"))
       
        pickle_path = os.path.join(directory, 'input_list')
        if not os.path.exists(pickle_path):
            os.makedirs(pickle_path)
        pickle.dump(self.model.input_list, open(os.path.join(pickle_path,'input_list.pickle'), "wb"))
        
           
    @classmethod
    def from_file(cls, directory, **kwargs):
        """ Load a model from file
        
        parameters
        -----------
        directory: (string, path)
                A string for the path to a directory in whcih to store the model. if the directory 
                doesn't exist, it will be created.
                
        ** kwargs: optional kwargs for keras.model.load_model
        
        returns
        ---------
        
        cls: An instantiation of the concrete subclass being loaded.
        
        """
        model_path = os.path.join(directory, 'model')
        pickle_path = os.path.join(directory, 'processors', 'processors.pickle')
        stru_pickle_path = os.path.join(directory, 'stru_processors', 'stru_processors.pickle')
        tok_path = os.path.join(directory, 'extra_tokenizers', 'tokenizers.pickle')
        input_path = os.path.join(directory, 'input_list', 'input_list.pickle')
        
        model = keras.models.load_model(model_path,**kwargs)
        processors = pickle.load(open(pickle_path, "rb" ))
        stru_processors = pickle.load(open(stru_pickle_path, "rb"))
        toks = pickle.load(open(tok_path, "rb"))
        if os.path.exists(input_path):
            input_list = pickle.load(open(input_path, "rb"))
            model.input_list = input_list
        
        return cls(model, processors=processors, stru_processors=stru_processors, extra_tokenizers=toks, load=True)

#%%
# ----------------------------------------------------------------------------------------
class FusionModel(tf.keras.Model, ABC):
    """ Model for classifying X-cell-receptors with V/J and sequence info"""
    
    def __init__(self,
                 seq_encoders,
                 str_encoders,
                 extra_encoders,
                 predictor,
                 input_list=None,
                 **kwargs
                ):
        super(FusionModel,self).__init__(**kwargs)
        self.seq_encoders = seq_encoders
        self.str_encoders = str_encoders
        self.extra_encoders = extra_encoders
        self.predictor = predictor
        if input_list is None:
            input_list = sorted(list(seq_encoders.keys()))+sorted(list(str_encoders.keys()))+sorted(list(extra_encoders.keys()))
        self._set_input_list(input_list)
        self.z_concat = keras.layers.Concatenate(name='z_concat')

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
    
    def _set_input_list(self,input_list):
        self.input_list = input_list
        
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
                if isinstance(x_e, dict):
                    x_e = x_e['mu']
                vals.append(x_e)
            elif sel in list(self.str_encoders.keys()):
                try:
                    x_s = self.str_encoders[sel](inputs[sel],training=training)
                except:
                    raise RuntimeError('check structure encoder')
                if isinstance(x_s, dict):
                    x_s = x_s['mu']
                vals.append(x_s)
            elif sel in list(self.extra_encoders.keys()):
                vals.append(self.extra_encoders[sel](inputs[sel],training=training))
            else:
                raise RuntimeError(f"unknown selection key passed: {sel}")
        
        if len(vals)>1:
            z = self.z_concat(vals)
        else:
            z = vals[0]
        return z  

#%%
# ----------------------------------------------------------------------------------------
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
        self.predictor = FusionNet(gated_fuison)
        
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

# ----------------------------------------------------------------------------------------
class FusionNet(tf.keras.Model, ABC):

    def __init__(self, options):
        super(FusionNet, self).__init__()
        self.fusion = BilinearFusion(skip=options['skip'], 
                                        gate1=options['seq_gate'], 
                                        gate2=options['stru_gate'], 
                                        gated_fusion=options['gated_fusion'],
                                        dim1=options['seq_dim'], 
                                        dim2=options['stru_dim'], 
                                        scale_dim1=options['seq_scale'], 
                                        scale_dim2=options['stru_scale'], 
                                        mmhid=options['mmhid'], 
                                        dropout_rate=options['dropout_rate'])
        self.classifier = tf.keras.layers.Dense(options['label_dim'], activation=options['activation'])


    def call(self, seq_stru_features, training=False):        
        features = self.fusion(seq_stru_features)
        Y_prob = self.classifier(features)
        return Y_prob

# ----------------------------------------------------------------------------------------
class BilinearFusion(tf.keras.Model, ABC):
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

#%%
# ----------------------------------------------------------------------------------------
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
        self.predictor = FusionNetMultiTask(gated_fuison)
        
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
        output = self.predictor([z_seq,z_stru], training=training)
        return output

        
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

# ----------------------------------------------------------------------------------------
class FusionNetMultiTask(tf.keras.Model, ABC):

    def __init__(self, options):
        super(FusionNetMultiTask, self).__init__()
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
        # self.output1.add(tf.keras.layers.Dense(16,activation='relu'))
        self.output1.add(tf.keras.layers.Dense(options['label_dim'], activation=options['activation'], name = 'output_1'))     

        self.output2 = tf.keras.Sequential()
        # self.output2.add(tf.keras.layers.Dense(16,activation='relu'))
        # self.output2.add(tf.keras.layers.Dense(8,activation='relu'))
        self.output2.add(tf.keras.layers.Dense(6, activation='softmax', name = 'output_2'))


    def call(self, seq_stru_features, training=False):        
        features = self.fusion(seq_stru_features)
        Y_reg = self.output1(features)
        Y_prob = self.output2(features)
        return [Y_reg, Y_prob]

# ----------------------------------------------------------------------------------------
class BilinearFusionMultiTask(tf.keras.Model, ABC):
    # we do not offer a bilinear transformation to the incoming data y = x1Ax2+b
    def __init__(self, skip=True,
                 gate1=True, gate2=True, gated_fusion=True,
                 dim1=32, dim2=32, 
                 scale_dim1=1, scale_dim2=1, 
                 mmhid=64, dropout_rate=0.25):
        super(BilinearFusionMultiTask, self).__init__()
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
        self.encoder2.add(tf.keras.layers.Dense(16,activation='relu'))
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