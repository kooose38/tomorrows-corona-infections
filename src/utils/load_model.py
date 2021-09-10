from src.modeling.lstm import LSTM 
import torch 

def load_net():
    HIDDEN_DIM = 128
    net = LSTM(HIDDEN_DIM)
    net.load_state_dict(torch.load("weights/net1500.pth", map_location={"cuda:0": "cpu"}))
    net.eval()
    return net
