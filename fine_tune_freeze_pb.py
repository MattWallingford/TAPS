from options import options
import torchvision.models as models
import torchvision
import os
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from utils import AverageMeter
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from models.freeze_net import resnet50, resnet101
import datasets
from datasets import Scale
import timm


def train(train_loader, model, criterion, optimizer, device, opts, epoch):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.train()
    # switch to train mode
    for i, (x, label) in enumerate(train_loader):
        # measure data loading time
        x = x.to(device)
        label = label.to(device)

        # compute output
        output = model(x)
        if epoch > opts.args.warmup_epochs:
            loss = opts.args.lam * sum([abs(i) for i in model.module.getIndicators()])/52 + criterion(output, label)
        else:
            loss = criterion(output, label)
        # measure accuracy and record loss
        acc1 = accuracy(output, label, topk=(1,))
        losses.update(loss.item(), x.size(0))
        top1.update(acc1[0], x.size(0))

        # compute gradient and do SGD step
        loss.backward()
        if i % 4 == 0:
            optimizer.step()
            optimizer.zero_grad()
    return losses.avg, 100 - top1.avg.item()

def eval(val_loader, model, criterion, device):
    '''
    evaluate the initialization, currently including training loss and training error
    TODO: adding other evaluation metrics
    '''

    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    model.eval()
    with torch.no_grad():
        for i, (x, label) in enumerate(val_loader):
            x = x.to(device)
            label = label.to(device)

            # compute output
            output = model(x)
            loss = criterion(output, label)

            # measure accuracy and record loss
            acc1 = accuracy(output, label, topk=(1,))
            losses.update(loss.item(), x.size(0))
            top1.update(acc1[0], x.size(0))

    return losses.avg, 100 - top1.avg.item()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch, init_lr, lr_decay_epoch):
    """Step based learning rate schedule sets the learning rate to the initial LR decayed by 10 in a given schedule"""

    lr_step_counter = 0
    for epoch_step in lr_decay_epoch:
        if epoch > epoch_step:
            lr_step_counter += 1

    lr = init_lr * (0.1 ** lr_step_counter)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def get_pytorch_model(num_classes, opts:options):
    model = models.__dict__[opts.args.arch](pretrained = opts.args.pretrained)
    embedding_dim = model.fc.in_features
    model.fc = nn.Linear(embedding_dim, num_classes)
    return model


def _finetune(model, train_loader, val_loader, opts: options):
    '''finetune a model on a dataset with given hyperparameters

    :param model: initialized model with cuda device configured
    :param train_dataset:
    :param val_dataset:
    :param kwargs:

    :return: validation result after the final epoch of training
    '''

    arch = opts.args.arch
    optimizer = opts.args.optimizer
    
    epochs = opts.args.epochs
    momentum = opts.args.momentum
    init_lr = opts.args.lr
    wd = opts.args.wd
    gpu = opts.args.gpu
    ngpu = 1
    save_frequency = opts.args.save_frequency

    experiment_path = os.path.join(opts.args.result_path, opts.args.experiment_name)
    if opts.args.result_path and not os.path.exists(opts.args.result_path):
        os.makedirs(opts.args.result_path)

    if os.path.exists(experiment_path):
        print(opts.args.experiment_name + ' already exists. Skipping.')
        return


    if not(os.path.exists(experiment_path)):
        os.makedirs(experiment_path)

    writer = SummaryWriter(experiment_path)
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    params = model.parameters()
    optimizer = torch.optim.AdamW(params, init_lr)

    best_val_top1_err = 100
    scheduler = CosineAnnealingLR(optimizer, T_max = epochs)
    val_errs = []
    for epoch in tqdm(range(0, epochs + 1)):
        train_loss, train_err = train(train_loader, model, criterion, optimizer, device, opts, epoch)
        if epoch % opts.args.eval_epochs == 0:
            # evaluate the performance of initialization
            val_loss, val_err = eval(val_loader, model, criterion, device)
            val_errs.append(val_err)
            writer.add_scalar('Validation Error', val_err, epoch)
            is_best = val_err <= best_val_top1_err
            best_val_top1_err = min(val_err, best_val_top1_err)
            state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': model.module.state_dict() if ngpu > 1 else model.state_dict()
            }

            if is_best:
                best_model_path = ('%s/model_best.pth' % experiment_path)
                torch.save(state, best_model_path)

            opt_state = {
                'optimizer': optimizer.state_dict(),
            }
            val_path = os.path.join(experiment_path, 'val_err')
            np.save(val_path, val_errs)
        scheduler.step()
        writer.add_scalar('Percent Weights Activated', torch.mean((torch.tensor(model.module.getIndicators()) >= .1).float()), epoch)
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('Training Loss', train_loss, epoch)
        writer.add_scalar('Training Error', train_err, epoch)



if __name__ == "__main__":
    opts = options()
    query_expr = opts.args.dataset
    if opts.args.subsample:
        query_expr += '.subuniform(' + str(opts.args.subsample) + ')'
    if opts.args.Vit:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    test_transform = transforms.Compose([
        Scale((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    train_transform = transforms.Compose([
        Scale((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    train_path = '../../' + opts.args.dataset + '/train'
    test_path = '../../' + opts.args.dataset + '/test'
    train_dataset = torchvision.datasets.ImageFolder(train_path, transform = train_transform)
    val_dataset = torchvision.datasets.ImageFolder(test_path, transform = test_transform)
    num_classes = len(train_dataset.classes)
    print('number of classes: ', num_classes)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opts.args.batch_size, shuffle=True,
        num_workers=opts.args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opts.args.batch_size, shuffle=True,
        num_workers=opts.args.workers, pin_memory=True)
        
 
    if opts.args.Vit:
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes= num_classes)
    else:
        print('loading r50')
        model = resnet50()
        z = 'resnet50.pth'
        state_dict = torch.load(z)

        del state_dict['fc.weight']
        del state_dict['fc.bias']
        model.load_state_dict(state_dict, strict = False)
        embedding_dim = model.fc.in_features
        model.fc = nn.Linear(embedding_dim, num_classes)

    device = torch.device(opts.args.gpu)
    model = model.to(device)
    if opts.args.multi_gpu:
        model = nn.DataParallel(model)
    _finetune(model, train_loader, val_loader, opts)
    opts.log_settings()