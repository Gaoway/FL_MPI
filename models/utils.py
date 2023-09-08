from models.dnn_rec import DNN_REC
from models.wide_deep import WideAndDeepModel
from config import cfg
import torch


def create_model_instance(model_type, field_dims=None):
    torch.manual_seed(2024)
    if model_type == 'dnn_rec':
        return DNN_REC(field_dims)
    elif model_type == 'wide&deep':
        return WideAndDeepModel(field_dims, cfg['embed_dim'], cfg['mlp_dims'], cfg['dropout'])
