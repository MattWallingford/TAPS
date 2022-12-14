from options import options
import torchvision.models as models
import os
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from utils import AverageMeter, get_pretrained_weights, load_model, load_joint_model, create_transforms
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from models.multihead_net import resnet34
from samplers.DomainNet import MultiDomainSampler
import torchvision

#Names and number of classes for all datasets 
DATA_ROOT = '../DomainNet/'
DATASET_NAMES = ['sketch', 'clipart', 'infograph', 'painting', 'quickdraw', 'real']
CLASS_SIZES = [345, 345, 345, 345, 345, 345]


def train_standard_multi(train_loader, model, criterion, optimizer, device, opts, epoch, num_tasks):
    losses = []
    top1 = []
    for i in range(num_tasks):
        losses.append(AverageMeter('Loss', ':.4e'))
        top1.append(AverageMeter('Acc@1', ':6.2f'))

    model.train()
    for x, (label, task) in train_loader:
        # measure data loading time
        x = x.to(device)
        label = label.to(device)

        for i in range(num_tasks):
            task_data = x[task == i]
            task_labels = label[task == i]
            task_labels = task_labels.long()
            output = model(task_data, i)
            loss = criterion(output, task_labels)
            loss.backward()
            acc1 = accuracy(output, task_labels, topk=(1,))
            losses[i].update(loss.item(), x.size(0))
            top1[i].update(acc1[0], x.size(0))

        #Update after accumulating gradients for all tasks
        optimizer.step()
        optimizer.zero_grad()

    #Calculate Per Task Accuracy 
    err = np.array([100-acc.avg for acc in top1])
    losses_avg = np.array([loss.avg for loss in losses]).mean()
    return losses_avg, err

def test_standard_multi(val_loader, model, criterion, device, num_tasks):
    losses = []
    top1 = []
    for i in range(num_tasks):
        losses.append(AverageMeter('Loss', ':.4e'))
        top1.append(AverageMeter('Acc@1', ':6.2f'))

    model.eval()
    for x, (label, task) in val_loader:
        # measure data loading time
        x = x.to(device)
        label = label.to(device)

        for i in range(num_tasks):
            task_data = x[task == i]
            task_labels = label[task == i]
            task_labels = task_labels.long()
            output = model(task_data, i)
            loss = criterion(output, task_labels)
            acc1 = accuracy(output, task_labels, topk=(1,))
            losses[i].update(loss.item(), x.size(0))
            top1[i].update(acc1[0], x.size(0))

    err = np.array([100-acc.avg for acc in top1])
    losses_avg = np.array([loss.avg for loss in losses]).mean()
    return losses_avg, err

def eval_multi(val_loader, model, criterion, device, task_partitions):
    top1 = []
    losses = []
    
    for i in range(len(task_partitions-1)):
        top1.append(AverageMeter('Acc@1_' + str(i), ':6.2f'))
        losses.append(AverageMeter('loss_' + str(i), ':6.2f'))
    model.eval()
    with torch.no_grad():
        for i, (x, label) in enumerate(val_loader):
            x = x.to(device)
            label = label.to(device)

            for i in range(len(task_partitions)-1):
                task_data = x[(label >= task_partitions[i]) & (label < task_partitions[i+1])]
                task_labels = label[(label >= task_partitions[i]) & (label < task_partitions[i+1])] -  task_partitions[i]
                # compute output
                task_labels = task_labels.long()
                output = model(task_data, i)
                loss = criterion(output, task_labels)

                # measure accuracy and record loss
                acc1 = accuracy(output, task_labels, topk=(1,))
                losses[i].update(loss.item(), task_data.size(0))
                top1[i].update(acc1[0], task_data.size(0))
        
    err = np.array([100-acc.avg for acc in top1])
    losses_avg = np.array([loss.avg for loss in losses]).mean()
    return losses_avg, err


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



def _finetune(model, train_loader, val_loader, opts: options, task_num):

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
        print(opts.args.experiment_name + ' already exists. Overwriting.')


    if not(os.path.exists(experiment_path)):
        os.makedirs(experiment_path)

    writer = SummaryWriter(experiment_path)

    criterion = nn.CrossEntropyLoss().cuda(gpu)

    use_nesterov = True if opts.args.optimizer == 'nag' else False


    optimizer = torch.optim.SGD(model.parameters(), init_lr,
                                momentum=momentum,
                                weight_decay=wd,
                                nesterov=use_nesterov)

    best_val_top1_err = 100
    scheduler = CosineAnnealingLR(optimizer, T_max = epochs)
    val_errs_total = []
    for epoch in tqdm(range(0, epochs + 1)):
        train_loss, train_err = train_standard_multi(train_loader, model, criterion, optimizer, device, opts, epoch, task_num)
        if epoch % opts.args.eval_epochs == 0:
            val_loss, val_errs = test_standard_multi(val_loader, model, criterion, device, task_num)
            val_errs_total.append(val_errs)
            for j, i in enumerate(DATASET_NAMES):
                writer.add_scalar('Validation Error_' + i, val_errs[j], epoch)

            state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': model.module.state_dict() if opts.args.multi_gpu else model.state_dict()
            }

            opt_state = {
                'optimizer': optimizer.state_dict(),
            }
            save_model_path = '%s/model-%04d.pth' % (experiment_path, epoch)
            save_opt_path = '%s/opt-%04d.pth' % (experiment_path, epoch)
            torch.save(state, save_model_path)
            torch.save(opt_state, save_opt_path)
            val_path = os.path.join(experiment_path, 'val_err')
            np.save(val_path, val_errs_total)
        scheduler.step()
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('Training Loss', train_loss, epoch)
        for j, i in enumerate(DATASET_NAMES):
            writer.add_scalar('Training Error_' + i, train_err[j], epoch)


if __name__ == "__main__":
    opts = options()   
    train_transform, test_transform = create_transforms(opts)
    train_sets = []
    val_sets = []
    dataset_names = os.listdir(opts.args.dataset)
    num_classes = []

    #Concatenate all datasets together
    for j, dataset_name in enumerate(dataset_names):
        train_path = os.path.join(opts.args.dataset, dataset_name) + '/train'
        test_path = os.path.join(opts.args.dataset, dataset_name) + '/test'
        print(train_path)
        train_dataset = torchvision.datasets.ImageFolder(train_path, transform = train_transform)
        val_dataset = torchvision.datasets.ImageFolder(test_path, transform = test_transform)
        train_dataset.samples = [(x, (label, j)) for x, label in train_dataset.samples]
        val_dataset.samples = [(x, (label, j)) for x, label in val_dataset.samples]
        print(int(max(train_dataset.targets)+1))
        num_classes.append(int(max(train_dataset.targets)+1))
        train_sets.append(train_dataset)
        val_sets.append(val_dataset)

    idx_list_train = []
    curr = 0
    for i in train_sets:
        idx_list_train.append(list(np.arange(len(i)) + curr))
        curr += len(i)

    idx_list_val = []
    curr = 0
    for i in val_sets:
        idx_list_val.append(list(np.arange(len(i)) + curr))
        curr += len(i)

    train_samp = MultiDomainSampler(idx_list_train, batch_size = opts.args.batch_size, domain_names = np.arange(0,len(train_sets)), random_shuffle = True)
    val_samp = MultiDomainSampler(idx_list_val, batch_size = opts.args.batch_size, domain_names = np.arange(0,len(val_sets)), random_shuffle = True)


    train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(train_sets), 
        batch_sampler = train_samp, num_workers=opts.args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(val_sets), 
        batch_sampler = val_samp, num_workers=opts.args.workers, pin_memory=True)
    model = load_joint_model(opts.args.model_type)
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
    
    model.set_task_partitions(num_classes)
    device = torch.device(opts.args.gpu)
    model = model.to(device)
    if opts.args.multi_gpu:
        model = nn.DataParallel(model)
    _finetune(model, train_loader, val_loader, opts, len(DATASET_NAMES))