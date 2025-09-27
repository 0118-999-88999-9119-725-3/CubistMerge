import argparse
import timm
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import torch

from fvcore.nn import FlopCountAnalysis
import time

import tome
import cume

from datasets import build_dataset

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    
    parser.add_argument('-l', default=1, type=int, help='Integer value for layer to apply token merging (for CuMe only)')
    parser.add_argument('-rh', type=int, help='Int value for rh parameter (for CuMe only)')
    parser.add_argument('-rw', type=int, help='Int value for rw parameter (for CuMe only)')
    parser.add_argument('-r', type=int, help='Int value for r parameter (for ToMe only)')
    parser.add_argument('-m', choices=['cume', 'tome'], 
                       help='Method choice: cume, tome')

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--data_path', default='/data2/IMAGENET-UNCROPPED/', type=str,
                        help='dataset path')
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    return parser

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

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model, transform):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.eval()

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')


    first_batch = True
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()


            output = model(images)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            if not first_batch:
                batch_time.update(time.time() - end)
            else:
                first_batch = False
            end = time.time()


            if i % (20) == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    return top1.avg


def calc_flops(val_loader, model):
    model.eval()
    first_batch = True
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            flops = FlopCountAnalysis(model,images)
            print(flops.total() / 1e9)
            break


def main(args):
    model_name = "deit_base_patch16_224"
    model = timm.create_model(model_name, pretrained=True)

    input_size = model.default_cfg["input_size"][1]
    transform = transforms.Compose([
        transforms.Resize(int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(model.default_cfg["mean"], model.default_cfg["std"]),
    ])

    dataset_val, _ = build_dataset(transform, is_train=False, args=args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=10,
        pin_memory=True,
        drop_last=False
    )

    if args.m == "tome":
        tome.patch.timm(model)
        model.r = args.r
    if args.m == "cume":
        cume.patch.timm(model, args.l, args.rh, args.rw)

    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of params:', n_parameters)
    model = model.cuda()

    calc_flops(data_loader_val, model)
    validate(data_loader_val, model, transform)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dynamic evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.m == "tome" and args.r is None:
        parser.error("Must specify r for ToMe")
    if args.m == "cume" and (args.rh is None or args.rw is None or args.l is None):
        parser.error("Must specify rh and rw for CuMe")
    main(args)