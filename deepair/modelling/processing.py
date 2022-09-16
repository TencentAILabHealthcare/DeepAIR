import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from Bio.Data.IUPACData import protein_letters

#%%
class  AASeqProcessorTransformer:

    """ A simple general processor for sequences of amino acids

    parameters
    ------------
    max_len: (int)
            A maximum length of the output sequences - sequences will be padded with 
            0's such that alll sequences are the same length.

    extra_chars: (string, optional, default ='')
            A single string containing any characters that should be expected in the AA
            strings that will be passed as input. 

    """

    def __init__(self, 
                tokenizer, 
                max_len, 
                add_special_tokens=True, 
                padding = 'max_length', 
                truncation = True, 
                return_tensors = 'np',
                isIDOnly = True
                ):

        self.max_length = max_len
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        self.isIDOnly = isIDOnly         

    def transform_seqs(self,x):
        """ convert AA strings to integer array
        parameters
        -----------
        x: list
            a list of AA strings
            
        returns
        --------
        seqs: np.array
                numpy array of integer sequences of fixed length dim (len(x), self.max_len)
        
        """
        seqs = self.tokenizer.batch_encode_plus(x, 
                                             add_special_tokens = self.add_special_tokens, 
                                             padding = self.padding, 
                                             truncation = self.truncation, 
                                             max_length = self.max_length, 
                                             return_tensors = self.return_tensors
                                             )
        if self.isIDOnly:
            return seqs['input_ids']
        else:
            return seqs

    def transform(self, x, **kwargs):
        """ convert AA strings to integer array
        parameters
        -----------
        x: list
            a list of AA strings
            
        returns
        --------
        transformed: dict
            dict with at least the single key 'seqs', with values being converted AA sequences.
            Subclasses may provide further key-value pairs here.
        
        """
        # print('-'*30)
        # print(self.transform_seqs(x).shape)
        # print(self.transform_seqs(x))
        # print('-'*30)
        return {'seqs': self.transform_seqs(x)}

    
    def _get_feature_dim(self):
        """ Returns dimension of the tokenizer, i.e number of characters it encodes """
        return len(self.tokenizer.vocab)
    
    @property
    def feature_dim(self):
        """ number of features at each seq position """
        return self._get_feature_dim()
    
    @property
    def vocab_size(self):
        """ number of tokens in tokenizer """
        return len(self.tokenizer.vocab)