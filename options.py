import os 
import random 
import warnings
import argparse

class options():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, help = 'dataset name')
        parser.add_argument('--lam', default=1, type = float, help = 'sparsity factor')
        parser.add_argument('--warmup_epochs', default=2, type = float, help = 'Number of epochs before regularizing')
        parser.add_argument('--score_lr', default=.01, type = float, help = '')
        parser.add_argument('--init_path', type = str)
        parser.add_argument('--multi_gpu', action = 'store_true')
        parser.add_argument('--train_layers', action = 'store_true')
        parser.add_argument('--Vit', default = False, type = bool)
        parser.add_argument('--joint', default = False, type = bool)
        parser.add_argument('--arch', default = 'resnet50')
        parser.add_argument('--eval_epochs', default = 1, type = int)
        parser.add_argument('--workers', default = 8, type = int)
        parser.add_argument('--epochs', default = 30, type = int)
        parser.add_argument('--batch_size', default = 32, type = int)
        parser.add_argument('--dtype', type=str, default='fp32')
        parser.add_argument('--optimizer', type=str, default = 'nag')
        parser.add_argument('--lr', '--learning_rate', default = .01, type = float)
        parser.add_argument('--momentum', type = float, default=.9)
        parser.add_argument('--wd', type = float, default=0)
        parser.add_argument('--gpu', type = int, default=0)
        parser.add_argument('--save-frequency', type = int, default=10)
        parser.add_argument('--result_path', type = str, default='')
        parser.add_argument('--model_path', type = str, default='')
        parser.add_argument('--experiment_name', type = str, default='')
        parser.add_argument('--resize', type = int, default=256)
        parser.add_argument('--input_size', type = int, default=224)
        parser.add_argument('--data-aug', type = str, default='rrcrop')
        parser.add_argument('--model_type', type = str, default='resnet34')
        parser.add_argument('--cropped', action = 'store_true')
        args = parser.parse_args()
        self.args = args

    def log_settings(self):
        write_path = os.path.join(self.args.result_path, self.args.experiment_name)

        f = open(os.path.join(write_path, "settings.txt"), "w")
        settings = str(self.args)
        strings = ['Namespace', '(', ')']
        replacements = ['','',', ']
        for string, replacement in zip(strings, replacements):
            settings = settings.replace(string, replacement)
        f.write(settings)
        f.close()