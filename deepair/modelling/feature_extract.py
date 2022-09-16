import os
from abc import ABC

#%%
class FeatureExtractiongModel(ABC):
    """ Base class for XCR Models, and associated processors
    
    Base class providing useful tools for handling multiple inputs and 
    passing those inputs into a BaseXCRModel. Evaluators can be added so that a single
    evaluation call can contain multiple results.
    Save and load functionality is provided allowing model and processors to be easily 
    saved and loaded together as a single object.
    
    Input can be passed as simple strings and gene names, rather than numerical representations.
    Typically, class methods expecting 'inputs' arguments, expect a dictionary with keys, 
    for example:
    {'ACDR3': array of strings,
     'BCDR3': array of strings,
     'V_beta': array of strings.......}
    
    
    parameters
    -----------
    model: (tcrai.modelling.base_models.BaseXCRModel subclass) 
            A BaseXCRModel subclass model.
    processors: (dict of tcrai.modelling.processors Processor) 
            A dict with form {name of expected model input : a processor object}
    extra_tokenizers: (dict of keras.tokenizers) 
            A dict with form  {name of expected model input : a keras tokenizer}
  
    """
    
    def __init__(self, seq_processors = dict(), seq_encoders = dict(), input_list=None, **kwargs):
        super(FeatureExtractiongModel, self).__init__(**kwargs)
        self.seq_processors = seq_processors
        self.seq_encoders = seq_encoders
        if input_list is None:
            input_list = sorted(list(seq_encoders.keys()))
        self.input_list = input_list
    
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
        for key, val in self.seq_processors.items():
            pro_dict[key] = val.transform(inputs[key])['seqs']
        return pro_dict

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
        
        vals = dict()
        for sel in selection:
            if sel in list(self.seq_encoders.keys()):
                try:
                    x_e = self.seq_encoders[sel](inputs[sel], training=training)
                except:
                    raise RuntimeError('check sequence encoder')
                vals[sel] = x_e
            else:
                raise RuntimeError(f"unknown selection key passed: {sel}")
        return vals
    
    def run(self, inputs, training=None):
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

        pred_input = self.process_seqs(inputs)
        preds = self._calculate_z(pred_input, training=training)
        return preds