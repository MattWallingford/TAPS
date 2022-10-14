import torchvision
import torch
import torchvision.transforms as transforms
from datasets import Scale
from models.taps_net import resnet50, resnet101, resnet34
from models.multihead_net import resnet50 as mh_resnet50, resnet34 as mh_resnet34, resnet101 as mh_resnet101
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

def get_pretrained_weights(model_type):
    "Creates pytorch .pth file"
    if model_type == 'resnet34':
        model = torchvision.models.resnet34(pretrained = True)
        torch.save(model.state_dict(), str(model_type) + '.pth')
    if model_type == 'resnet50':
        model = torchvision.models.resnet50(pretrained = True)
        torch.save(model.state_dict(), str(model_type) + '.pth')
    if model_type == 'resnet101':
        model = torchvision.models.resnet101(pretrained = True)
        torch.save(model.state_dict(), str(model_type) + '.pth')

def load_model(model_type):
    if model_type == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes= num_classes)
    elif model_type == 'resnet34':
        model = resnet34()
    elif model_type == 'resnet50':
        model = resnet50()
    elif model_type == 'resnet101':
        model = resnet101()
    else:
        raise Exception('Model type, {}, not an available option. Ending Training run.').format(model_type)
    return model
    


def load_joint_model(model_type):
    if model_type == 'resnet34':
        model = mh_resnet34()
    elif model_type == 'resnet50':
        model = mh_resnet50()
    elif model_type == 'resnet101':
        model = mh_resnet101()
    else:
        raise Exception('Model type, {}, not an available option. Ending Training run.').format(model_type)
    return model

def create_transforms(opts):
    if opts.args.model_type == 'vit':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if opts.args.cropped:
        test_transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            normalize,
        ])
        train_transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else: 
        test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        ])

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    return train_transform, test_transform