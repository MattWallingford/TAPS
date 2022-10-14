from options import options
import torchvision.models as models
import torchvision
import os
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from utils import AverageMeter, get_pretrained_weights, load_model, create_transforms
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from models.taps_net import resnet50, resnet101, resnet34
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
            indicators = model.module.getIndicators() if opts.args.multi_gpu else model.getIndicators()
            loss = opts.args.lam * sum([abs(i) for i in indicators])/52 + criterion(output, label)
        else:
            loss = criterion(output, label)
        # measure accuracy and record loss
        acc1 = accuracy(output, label, topk=(1,))
        losses.update(loss.item(), x.size(0))
        top1.update(acc1[0], x.size(0))

        # compute gradient and do SGD step
        loss.backward()
        if i % 8 == 0:
            optimizer.step()
            optimizer.zero_grad()
    return losses.avg, 100 - top1.avg.item()

def eval(val_loader, model, criterion, device):
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


def _finetune(model, train_loader, val_loader, opts: options):
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
    optimizer = torch.optim.SGD(params, init_lr,
                                momentum=momentum,
                                weight_decay=wd)
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
            'state_dict': model.module.state_dict() if opts.args.multi_gpu else model.state_dict()
            }
            if is_best:
                best_model_path = ('%s/model_best.pth' % experiment_path)
                torch.save(state, best_model_path)

            opt_state = {
                'optimizer': optimizer.state_dict(),
            }
            val_path = os.path.join(experiment_path, 'val_err')
            np.save(val_path, val_errs)
        
        indicators = model.module.getIndicators() if opts.args.multi_gpu else model.getIndicators()
        scheduler.step()
        writer.add_scalar('Percent Weights Activated', torch.mean((torch.tensor(indicators) >= .1).float()), epoch)
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('Training Loss', train_loss, epoch)
        writer.add_scalar('Training Error', train_err, epoch)



if __name__ == "__main__":
    opts = options()
    train_transform, test_transform = create_transforms(opts)
    train_path = opts.args.dataset + '/train'
    test_path = opts.args.dataset + '/test'
    train_dataset = torchvision.datasets.ImageFolder(train_path, transform = train_transform)
    val_dataset = torchvision.datasets.ImageFolder(test_path, transform = test_transform)
    num_classes = len(train_dataset.classes)
    print('Number of classes: ', num_classes)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opts.args.batch_size, shuffle=True,
        num_workers=opts.args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opts.args.batch_size, shuffle=True,
        num_workers=opts.args.workers, pin_memory=True)
        
    #Initialize pretrained model
    model = load_model(opts.args.model_type)
    if opts.args.model_path:
        print('loading model from: {}'.format(opts.args.model_path))
        model_path = opts.args.model_path
        state_dict = torch.load(model_path)
        del state_dict['fc.weight']
        del state_dict['fc.bias']

    else:
        print('Loading pytorch pretrained model')
        if not(os.path.exists(opts.args.model_type + '.pth')):
            get_pretrained_weights(opts.args.model_type)

        model_path = opts.args.model_type + '.pth'
        state_dict = torch.load(model_path)

    model.load_state_dict(state_dict, strict = False)
    embedding_dim = model.fc.in_features
    model.fc = nn.Linear(embedding_dim, num_classes)
    if opts.args.multi_gpu:
        model = nn.DataParallel(model)

    device = torch.device(opts.args.gpu)
    model = model.to(device)
    _finetune(model, train_loader, val_loader, opts)
    opts.log_settings()
