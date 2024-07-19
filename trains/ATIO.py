"""
ATIO -- All Trains in One
"""
from .multiTask import *
from .singleTask import *
from .missingTask import *

__all__ = ['ATIO']

class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            # single-task
            'tfn': TFN,
            'tfnn': TFNN,
            'lmf': LMF,
            'mfn': MFN,
            'ef_lstm': EF_LSTM,
            'lf_dnn': LF_DNN,
            'graph_mfn': Graph_MFN,
            'mult': MULT,
            'mctn': MCTN,
            'bert_mag':BERT_MAG,
            'misa': MISA,
            'mfm': MFM,
            'mmim': MMIM,
            # multi-task
            'mtfn': MTFN,
            'mlmf': MLMF,
            'mlf_dnn': MLF_DNN,
            'afn_ag_msa': AFN_AG_MSA,
            # missing-task
            'tfr_net': TFR_NET,
        }
    
    def getTrain(self, args):
        return self.TRAIN_MAP[args['model_name']](args)
