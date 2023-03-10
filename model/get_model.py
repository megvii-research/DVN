from .Model_RGB_NIR_Fusion import RGB_NIR_Fusion
from .Model_RGB_NIR_Recons import RGB_NIR_Recons
import torch
from torch import nn
from collections import OrderedDict

def get_model():
    model = RGB_NIR_Fusion()
    return model

def get_pretrain():
    model = RGB_NIR_Recons()
    return model

def load_model(model, mode):
    net = get_pretrain() if mode == 'Recons' else get_model()
    checkpoint = torch.load(model)
    net.cuda()
    try:
        net.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    return net


if __name__ == "__main__":
    from thop import profile
    from thop import clever_format

    model = get_model()
    # model = get_pretrain()
    input = torch.randn(1, 3, 128, 128)
    ir = torch.randn(1, 1, 128, 128)
    flops, params = profile(model, inputs=(input, ir))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)