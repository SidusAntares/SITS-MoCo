"""
This script is for time series classification task.
"""
import copy
import argparse
from tqdm import tqdm
from joblib import dump, load

import torch.optim
import torch.nn.functional as F

from dataset import PixelSetData
from torch.utils.data.sampler import WeightedRandomSampler
from timematch_utils import label_utils
from collections import Counter
from dataset import PixelSetData, create_evaluation_loaders
from timematch_utils.train_utils import bool_flag
from transforms import (
    Normalize,
    RandomSamplePixels,
    RandomSampleTimeSteps,
    ToTensor,
    AddPixelLabels
)
from torch.utils import data
import numpy as np
import pandas as pd
import random
from utils import *


import torch


class TimeMatchToUSCropsAdapter:
    def __init__(self, device):
        self.device = device

    def __call__(self, batch_dict):
        pixels = batch_dict['pixels']
        positions = batch_dict['positions']
        valid_pixels = batch_dict['valid_pixels']
        pixel_labels = batch_dict['pixel_labels']

        B, T, C, N = pixels.shape

        # 1. 展平 (此时还在 CPU)
        data_flat = pixels.permute(0, 3, 1, 2).reshape(-1, T, C)
        doy_flat = positions.unsqueeze(1).expand(-1, N, -1).reshape(-1, T)

        valid_flat = valid_pixels.permute(0, 2, 1).reshape(-1, T).bool()
        mask_flat = ~valid_flat

        # 2. 处理标签
        y_flat = pixel_labels.reshape(-1)
        has_valid_time = valid_flat.any(dim=1)

        IGNORE_INDEX = -100
        # 确保 IGNORE_INDEX 张量也在 CPU 上先创建，稍后一起移动
        y_flat = torch.where(has_valid_time, y_flat, torch.tensor(IGNORE_INDEX, dtype=y_flat.dtype))
        y_flat = y_flat.long()

        # 3. 构建 Tuple (此时全在 CPU)
        X_tuple = (
            data_flat,
            mask_flat,
            doy_flat,
            valid_flat.float()
        )

        # 【关键修复】将所有数据移动到指定设备 (GPU)
        if self.device is not None:
            X_tuple = tuple(t.to(self.device) if torch.is_tensor(t) else t for t in X_tuple)
            y_flat = y_flat.to(self.device)

        return X_tuple, y_flat

def train_epoch_with_adapter(model, optimizer, criterion, dict_dataloader, adapter, device, args):
    """
    专门用于处理 PixelSetData (Dict 格式) 的训练循环
    """
    losses = AverageMeter('Loss', ':.4e')
    model.train()

    with tqdm(enumerate(dict_dataloader), total=len(dict_dataloader), leave=True) as iterator:
        for idx, batch_dict in iterator:
            # 【核心步骤】使用 Adapter 将 Dict 转换为 (X_tuple, y)
            try:
                X, y = adapter(batch_dict)
            except Exception as e:
                print(f"Error adapting batch {idx}: {e}")
                continue

            # 此时 X 已经是 tuple, y 是 tensor，且都在 device 上 (Adapter 内部处理了 to(device))
            # 不需要再调用 recursive_todevice(X, device)，因为 Adapter 已经做了
            # 但为了保险，如果 Adapter 没做全，可以保留 y 的 check
            if y.device != device:
                y = y.to(device)

            optimizer.zero_grad()

            # 调用模型
            if args.use_doy:
                logits = model(X, use_doy=True)
            else:
                logits = model(X)

            out = F.log_softmax(logits, dim=-1)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            iterator.set_description(f"train loss={loss:.2f}")
            losses.update(loss.item(), y.size(0))

    return losses.avg


def test_epoch_with_adapter(model, criterion, dict_dataloader, adapter, device, args):
    """
    专门用于处理 PixelSetData (Dict 格式) 的验证/测试循环
    """
    losses = AverageMeter('Loss', ':.4e')
    model.eval()
    y_true_list = list()
    y_pred_list = list()

    with torch.no_grad():
        with tqdm(enumerate(dict_dataloader), total=len(dict_dataloader), leave=True) as iterator:
            for idx, batch_dict in iterator:
                # 【核心步骤】转换数据
                try:
                    X, y = adapter(batch_dict)
                except Exception as e:
                    continue

                if y.device != device:
                    y = y.to(device)
                # 在 logits = model(X, ...) 之前加入
                if idx == 0:  # 只打印第一个 batch
                    data, mask, doy, weight = X
                    print(f"\n--- Debug Batch Info ---")
                    print(f"Data shape: {data.shape}, Min: {data.min()}, Max: {data.max()}, Mean: {data.mean()}")
                    print(f"Mask shape: {mask.shape}, True count (padding): {mask.sum()}")
                    print(f"Labels (first 10): {y[:10]}")

                if args.use_doy:
                    logits = model(X, use_doy=True)
                else:
                    logits = model(X)

                out = F.log_softmax(logits, dim=-1)
                loss = criterion(out, y)

                iterator.set_description(f"test loss={loss:.2f}")
                losses.update(loss.item(), y.size(0))

                y_true_list.append(y.cpu())
                y_pred_list.append(out.argmax(-1).cpu())

    if len(y_true_list) == 0:
        return losses.avg, {}  # 防止空列表报错

    y_true = torch.cat(y_true_list).numpy()
    y_pred = torch.cat(y_pred_list).numpy()

    scores = accuracy(y_true, y_pred, args.nclasses + 1)
    return losses.avg, scores

def parse_args():
    parser = argparse.ArgumentParser(description='Train an evaluate time series deep learning models.')
    parser.add_argument('model', type=str, default="STNet",
                        help='select model architecture.')
    parser.add_argument('--use-doy', action='store_true',
                        help='whether to use doy pe with trsf')
    parser.add_argument('--rc', action='store_true',
                        help='whether to random choice the time series data')
    parser.add_argument('--interp', action='store_true',
                        help='whether to interplate the time series data')
    parser.add_argument('--useall', action='store_true',
                        help='whether to use all data for training')
    parser.add_argument('-n', '--num', default=3000, type=int,
                        help='number of labeled samples (training and validation) (default 3000)')
    parser.add_argument('-c', '--nclasses', type=int, default=20,
                        help='num of classes (default: 20)')
    parser.add_argument('--year', type=int, default=2019,
                        help='year of dataset')
    parser.add_argument('-seq', '--sequencelength', type=int, default=30,
                        help='Maximum length of time series data (default 70)')
    parser.add_argument('-j', '--workers', type=int, default=0,
                        help='number of CPU workers to load the next batch')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=512,
                        help='batch size (number of time series processed simultaneously)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3,
                        help='optimizer learning rate (default 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='optimizer weight_decay (default 1e-4)')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='warmup epochs')
    parser.add_argument('--schedule', default=None, nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by a ratio)')
    parser.add_argument('-l', '--logdir', type=str, default="./results",
                        help='logdir to store progress and models (defaults to ./results)')
    parser.add_argument('-s', '--suffix', default=None,
                        help='suffix to output_dir')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('-d', '--device', type=str, default=None,
                        help='torch.Device. either "cpu" or "cuda". default will check by torch.cuda.is_available() ')
    parser.add_argument('--pretrained', default=None, type=str,
                        help='path to pretrained checkpoint')
    parser.add_argument('--eval', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--freeze', action='store_true',
                        help='freeze pretrain model')

    # 以下都是timematch
    parser.add_argument(
        "--num_workers", default=8, type=int, help="Number of workers"
    )
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--balance_source", type=bool_flag, default=True, help='class balanced batches for source')
    parser.add_argument('--num_pixels', default=4096, type=int, help='Number of pixels to sample from the input sample')
    parser.add_argument('--seq_length', default=30, type=int,
                        help='Number of time steps to sample from the input sample')
    # 数据路径与域
    parser.add_argument('--data_root', default='/data/user/DBL/timematch_data', type=str,
                        help='Path to datasets root directory')
    # parser.add_argument('--data_root', default='/mnt/d/All_Documents/documents/ViT/dataset/timematch', type=str,
    #                     help='Path to datasets root directory')
    parser.add_argument('--source', default='france/30TXT/2017', type=str)
    parser.add_argument('--target', default='france/30TXT/2017', type=str)
    # 类别处理
    parser.add_argument('--combine_spring_and_winter', action='store_true')
    # 数据划分
    parser.add_argument('--num_folds', default=1, type=int)
    parser.add_argument("--val_ratio", default=0.1, type=float)
    parser.add_argument("--test_ratio", default=0.2, type=float)
    # 评估
    parser.add_argument('--sample_pixels_val', action='store_true')  # 布尔型开关参数（flag），它不需要传值，只需在命令行中出现或不出现该选项

    args = parser.parse_args()

    args.dataset = 'USCrops'


    modelname = args.model.lower()
    if args.interp and modelname in ['rf', 'tempcnn', 'lstm']:
        args.interp = True
    else:
        args.interp = False

    if args.interp:
        args.rc_str = 'Int'
    elif args.rc:
        args.rc_str = 'RC'
    else:
        args.rc_str = 'Pad'

    if args.use_doy:
        if args.suffix:
            args.suffix = 'doy_' + args.suffix
        else:
            args.suffix = 'doy'

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
# ================== put a patch ===============
    args.workers = args.num_workers
    args.num = args.num_pixels
# ==============================================

    return args

def get_data_loaders(splits, config, balance_source=True):

    strong_aug = transforms.Compose([
            RandomSamplePixels(config.num_pixels),
            RandomSampleTimeSteps(config.seq_length),
            Normalize(),
            ToTensor(),
            AddPixelLabels()
    ])

    source_dataset = PixelSetData(config.data_root, config.source,
            config.classes, strong_aug,
            indices=splits[config.source]['train'],)

    if balance_source:
        source_labels = source_dataset.get_labels()
        freq = Counter(source_labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_labels]
        sampler = WeightedRandomSampler(source_weights, len(source_labels))
        print("using balanced loader for source")
        source_loader = data.DataLoader(
            source_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            sampler=sampler,
            batch_size=config.batch_size,
            drop_last=True,
        )
    else:
        source_loader = data.DataLoader(
            source_dataset,
            num_workers=config.num_workers,
            pin_memory=True,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )
    print(f'size of source dataset: {len(source_dataset)} ({len(source_loader)} batches)')

    return source_loader

def create_train_val_test_folds(datasets, num_folds, num_indices, val_ratio=0.1, test_ratio=0.2):
    folds = []
    for _ in range(num_folds):
        splits = {}
        for dataset in datasets:
            if type(num_indices) == dict:
                indices = list(range(num_indices[dataset]))
            else:
                indices = list(range(num_indices))
            n = len(indices)
            n_test = int(test_ratio * n)
            n_val = int(val_ratio * n)
            n_train = n - n_test - n_val

            random.shuffle(indices)

            train_indices = set(indices[:n_train])
            val_indices = set(indices[n_train:n_train + n_val])
            test_indices = set(indices[-n_test:])
            assert set.intersection(train_indices, val_indices, test_indices) == set()
            assert len(train_indices) + len(val_indices) + len(test_indices) == n

            splits[dataset] = {'train': train_indices, 'val': val_indices, 'test': test_indices}
        folds.append(splits)
    return folds

def train(args):
    print("=> creating dataloader")
    random.seed(10)
    config = cfg = args
    source_classes = label_utils.get_classes(cfg.source.split('/')[0],
                                             combine_spring_and_winter=cfg.combine_spring_and_winter)
    source_data = PixelSetData(cfg.data_root, cfg.source, source_classes)
    labels, counts = np.unique(source_data.get_labels(), return_counts=True)
    source_classes = [source_classes[i] for i in labels[counts >= 200]]
    print('Using classes:', source_classes)
    cfg.classes = source_classes
    cfg.num_classes = len(source_classes)  # 可以覆盖该参数的默认设置
    # Randomly assign parcels to train/val/test
    indices = {config.source: len(source_data)}
    folds = create_train_val_test_folds([config.target], config.num_folds, indices, config.val_ratio,
                                        config.test_ratio)
    splits = folds[0]
    sample_pixels_val = config.sample_pixels_val
    val_loader_dict, test_loader_dict = create_evaluation_loaders(config.source, splits, config, sample_pixels_val)
    source_loader_dict = get_data_loaders(splits, config, config.balance_source)

    num_classes = cfg.num_classes
    args.nclasses = cfg.num_classes
    assert args.model not in ['rf', 'RF'] # 这两种模型的数据加载方式要求numpy数组
    ndims = 10

    device = torch.device(args.device)
    train_adapter = TimeMatchToUSCropsAdapter(device)
    val_adapter = TimeMatchToUSCropsAdapter(device)
    test_adapter = TimeMatchToUSCropsAdapter(device)

    print("=> creating model '{}'".format(args.model))
    model = get_model(args.model, ndims, num_classes, args.sequencelength, device)


    print(f"Initialized {model.modelname}: Total trainable parameters: {get_ntrainparams(model)}")
    model.apply(weight_init)
    finetune = False
    if args.pretrained is not None:
        finetune = True
        path = Path(args.pretrained).absolute().relative_to(Path(__file__).absolute().parent)
        print("=> loading checkpoint '{}'".format(str(path)))
        pretrain_model = torch.load(path)['model_state']
        model_dict = model.state_dict()
        if 'moco' in str(path.parts[-2]).lower():
            state_dict = {}
            for k in list(pretrain_model.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('encoder_q') and not k.startswith('encoder_q.decoder') and not k.startswith(
                        'encoder_q.classification') and not k.startswith('encoder_q.position_enc.pe'):
                    # remove prefix
                    state_dict[k[len("encoder_q."):]] = pretrain_model[k]  # module.
        else:
            state_dict = {k: v for k, v in pretrain_model.items() if
                          k in model_dict.keys() and 'decoder' not in k and 'position_enc.pe' not in k}

        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    if args.freeze:
        for name, param in model.named_parameters():
            if not name.startswith('decoder'):
                param.requires_grad = False

    if finetune:
        model.modelname = f'F_{path.parts[-2].split("_")[1][:2]}_{model.modelname}_R{args.num}_{args.rc_str}_{args.year}_Seed{args.seed}'
    elif args.useall:
        model.modelname = f'T_{model.modelname}_{args.rc_str}_{args.year}'
    else:
        model.modelname = f'T_{model.modelname}_R{args.num}_{args.rc_str}_{args.year}_Seed{args.seed}'

    logdir = Path(args.logdir) / model.modelname
    logdir.mkdir(parents=True, exist_ok=True)
    best_model_path = logdir / 'model_best.pth'
    print(f"Logging results to {logdir}")

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    if not args.eval:
        log = list()
        val_loss_min = np.inf
        print(f"Training {model.modelname}")
        for epoch in range(args.epochs):
            if args.warmup_epochs > 0:
                if epoch == 0:
                    lr = args.learning_rate * 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                elif epoch == args.warmup_epochs:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.learning_rate

            if args.schedule is not None:
                adjust_learning_rate(optimizer, epoch, args)
            train_loss = train_epoch_with_adapter(model, optimizer, criterion, source_loader_dict, train_adapter,
                                                  device, args)
            val_loss, scores = test_epoch_with_adapter(model, criterion, val_loader_dict, val_adapter, device, args)

            scores_msg = ", ".join(
                [f"{k}={v:.4f}" for (k, v) in scores.items() if k not in ['class_f1', 'confusion_matrix']])
            print(f"epoch {epoch + 1}: trainloss={train_loss:.4f}, valloss={val_loss:.4f} " + scores_msg)

            if val_loss < val_loss_min:
                not_improved_count = 0
                save(model, path=best_model_path, criterion=criterion)
                val_loss_min = val_loss
                print(f'lowest val loss in epoch {epoch + 1}\n')
            else:
                not_improved_count += 1

            scores["epoch"] = epoch + 1
            scores["trainloss"] = train_loss
            scores["testloss"] = val_loss
            log.append(scores)

            log_df = pd.DataFrame(log).set_index("epoch")
            log_df.to_csv(Path(logdir) / "trainlog.csv")

            if not_improved_count >= 10:
                print("\nValidation performance didn\'t improve for 10 epochs. Training stops.")
                break

        if epoch == args.epochs - 1:
            print(f"\n{args.epochs} epochs training finished.")

    # test
    # print('Restoring best model weights for testing...')
    # checkpoint = torch.load(best_model_path)
    # state_dict = {k: v for k, v in checkpoint['model_state'].items()}
    # criterion = checkpoint['criterion']
    # torch.save({'model_state': state_dict, 'criterion': criterion}, best_model_path)
    # model.load_state_dict(state_dict)

    test_loss, scores = test_epoch_with_adapter(model, criterion, test_loader_dict, test_adapter, device, args)
    scores_msg = ", ".join(
        [f"{k}={v:.4f}" for (k, v) in scores.items() if k not in ['class_f1', 'confusion_matrix']])
    print(f"Test results: \n\n {scores_msg}")

    scores['epoch'] = 'test'
    scores['testloss'] = test_loss
    conf_mat = scores.pop('confusion_matrix')
    class_f1 = scores.pop('class_f1')

    log_df = pd.DataFrame([scores]).set_index("epoch")
    log_df.to_csv(logdir / f"testlog.csv")
    np.save(logdir / f"test_conf_mat.npy", conf_mat)
    np.save(logdir / f"test_class_f1.npy", class_f1)

    return logdir


def train_epoch(model, optimizer, criterion, dataloader, device, args):
    losses = AverageMeter('Loss', ':.4e')
    model.train()
    with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
        for idx, (X, y) in iterator:
            X = recursive_todevice(X, device)
            y = y.to(device)

            optimizer.zero_grad()
            if args.use_doy:
                logits = model(X, use_doy=True)
            else:
                logits = model(X)
            out = F.log_softmax(logits, dim=-1)

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            iterator.set_description(f"train loss={loss:.2f}")

            losses.update(loss.item(), X[0].size(0))

    return losses.avg


def test_epoch(model, criterion, dataloader, device, args):
    losses = AverageMeter('Loss', ':.4e')
    model.eval()
    with torch.no_grad():
        y_true_list = list()
        y_pred_list = list()
        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, (X, y) in iterator:
                X = recursive_todevice(X, device)
                y = y.to(device)

                if args.use_doy:
                    logits = model(X, use_doy=True)
                else:
                    logits = model(X)
                out = F.log_softmax(logits, dim=-1)

                loss = criterion(out, y)
                iterator.set_description(f"test loss={loss:.2f}")
                losses.update(loss.item(), X[0].size(0))

                y_true_list.append(y)
                y_pred_list.append(out.argmax(-1))
    y_true = torch.cat(y_true_list).cpu().numpy()
    y_pred = torch.cat(y_pred_list).cpu().numpy()

    scores = accuracy(y_true, y_pred, args.nclasses + 1)

    return losses.avg, scores


def main():
    args = parse_args()
    years = [2019]
    for year in years:
        print(f' ===================== {year} ======================= ')
        args.year = year
        seeds = [111]
        print('seed in', seeds)
        for seed in seeds:
            args.seed = seed
            print(f'Seed = {args.seed} --------------- ')

            SEED = args.seed
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.deterministic = True

            logdir = train(args)
        overall_performance(str(logdir))


if __name__ == '__main__':
    main()
