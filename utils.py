import torchvision
import torch
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def select_layers(model, train_layers):
    all_layers = [b for a, b in model.named_modules() if (isinstance(b, nn.Sequential) and len(a.split('.')) == split_counter)]
    num_layers = len(all_layers)
    layers = all_layers[num_layers-train_layers:]
    layers.append(list(model.modules())[-1])
    print('training layers: {}'.format(layers))
    params = []
    for layer in layers:
        params += list(layer.parameters())
    return params



def load_pretrained_model(model_type):
    "Loads pretrained pytorch model"
    


def get_pretrained_weights(model_type):
    "Creates pytorch .pth file"
    if model_type == 'resnet34':
        model = torchvision.models.resnet34(pretrained = True)
        torch.save(model.state_dict(), str(model_type) + '.pth')
    if model_type == 'resnet50':
        model = torchvision.models.resnet50(pretrained = True)
        torch.save(model.state_dict(), str(model_type) + '.pth')
    