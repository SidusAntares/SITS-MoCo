"""
This script is for time series classification task.
"""
import copy
import argparse
import sys

from tqdm import tqdm


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
)
from torch.utils import data
import numpy as np
import pandas as pd
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import random
from utils import *
from datetime import datetime
import os

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

import torch


class TimeMatchToUSCropsAdapter:
    """
    将 Batch Dict 转换为模型输入 (X, y)。
    核心功能：处理全无效时间步样本，防止模型内除零错误 (NaN)。
    """

    def __init__(self, device):
        self.device = device
        self.warn_count = 0

    def __call__(self, batch_dict):
        pixels = batch_dict['pixels']  # [B, T, C, N]
        positions = batch_dict['positions']  # [B, N] (DOY)
        valid_pixels = batch_dict['valid_pixels']  # [B, N, T] (0/1)
        pixel_labels = batch_dict['label']  # [B, ]

        B, T, C, N = pixels.shape

        # 1. 展平维度: (B, N) -> Sample_Batch
        data_flat = pixels.permute(0, 3, 1, 2).reshape(-1, T, C)  # [S, T, C]
        doy_flat = positions.unsqueeze(1).expand(-1, N, -1).reshape(-1, T)  # [S, T]
        valid_flat = valid_pixels.permute(0, 2, 1).reshape(-1, T).bool()  # [S, T]
        mask_flat = ~valid_flat  # [S, T] (True=Invalid)
        y_flat = pixel_labels.reshape(-1).long()  # [S]

        # 构建输入元组
        X_tuple = (data_flat, mask_flat, doy_flat, valid_flat.float())

        # 6. 移至设备
        if self.device:
            X_tuple = tuple(t.to(self.device) if torch.is_tensor(t) else t for t in X_tuple)
            y_flat = y_flat.to(self.device)

        return X_tuple, y_flat


def train_epoch_with_adapter(model, optimizer, criterion, dict_dataloader, adapter, device, args):
    """
    专门用于处理 PixelSetData (Dict 格式) 的训练循环 (带强校验)
    """
    losses = AverageMeter('Loss', ':.4e')
    model.train()

    # 标记是否已经打印过调试信息
    debug_printed = False

    with tqdm(enumerate(dict_dataloader), total=len(dict_dataloader), leave=True) as iterator:
        for idx, batch_dict in iterator:
            try:
                X, y = adapter(batch_dict)
            except Exception as e:
                print(f"Error adapting batch {idx}: {e}")
                raise f"Error adapting batch {idx}: {e}"

            if y.device != device:
                y = y.to(device)

            optimizer.zero_grad()

            if args.use_doy:
                logits = model(X, use_doy=True)
            else:
                logits = model(X)


            out = F.log_softmax(logits, dim=-1)
            print("================train:")
            print(out.shape)
            print(y.shape)
            loss = criterion(out, y)

            loss.backward()

            # 梯度裁剪 (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            iterator.set_description(f"train loss={loss.item():.2f}")
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

                if args.use_doy:
                    logits = model(X, use_doy=True)
                else:
                    logits = model(X)

                out = F.log_softmax(logits, dim=-1)
                print("================test:")
                print(out.shape)
                print(y.shape)
                sys.exit()
                loss = criterion(out, y)

                iterator.set_description(f"test loss={loss:.2f}")
                losses.update(loss.item(), y.size(0))

                pred = out.argmax(-1).cpu()
                true = y.cpu()

                # 创建掩码：只保留标签 >= 0 的样本
                valid_mask = true >= 0

                y_true_list.append(true[valid_mask])
                y_pred_list.append(pred[valid_mask])

    if len(y_true_list) == 0:
        return losses.avg, {}  # 防止空列表报错

    y_true = torch.cat(y_true_list).numpy()
    y_pred = torch.cat(y_pred_list).numpy()

    # 3. 调用 accuracy 函数
    # 传入过滤后的数据
    # num_classes 传入 args.nclasses (即 12)，因为有效标签范围是 0-11
    # 原代码传的是 args.nclasses + 1，如果您的类别确实是 0-11，传 12 即可生成 12x12 矩阵
    # 如果原代码逻辑依赖 +1 (比如为了兼容某些特定索引)，请保持原样，但数据必须无负数
    scores = accuracy(y_pred, y_true, args.nclasses)
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
    parser.add_argument('-c', '--nclasses', type=int, default=20,
                        help='num of classes (default: 20)')
    parser.add_argument('--year', type=int, default=2019,
                        help='year of dataset')
    parser.add_argument('-seq', '--sequencelength', type=int, default=30,
                        help='Maximum length of time series data (default 70)')
    parser.add_argument('-j', '--workers', type=int, default=0,
                        help='number of CPU workers to load the next batch')
    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=500,
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

    # parser.add_argument('-n', '--num', default=10000, type=int,
    #                     help='number of labeled samples (training and validation) (default 3000)')
    parser.add_argument('-n', '--per', default=0.01, type=int,
                        help='percentage of labeled samples (training and validation) (default )')
    parser.add_argument('--seed', default=111, type=int,
                        help='seed')
    # 以下都是timematch
    parser.add_argument('--gpus', type=int, default=4,
                        help='Number of GPUs to use (0=CPU, 1=Single GPU, >=2=Multi-GPU DDP)')
    parser.add_argument(
        "--num_workers", default=8, type=int, help="Number of workers"
    )
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--balance_source", type=bool_flag, default=True, help='class balanced batches for source')
    parser.add_argument('--num_pixels', default=1, type=int, help='Number of pixels to sample from the input sample')
    parser.add_argument('--seq_length', default=30, type=int,
                        help='Number of time steps to sample from the input sample')
    # 数据路径与域
    parser.add_argument('--data_root', default='/mnt/d/All_Documents/documents/ViT/dataset/timematch', type=str,
                        help='Path to datasets root directory')
    # parser.add_argument('--data_root', default='/mnt/d/All_Documents/documents/ViT/dataset/timematch', type=str,
    #                     help='Path to datasets root directory')
    parser.add_argument('--source', default='france/31TCJ/2017', type=str)
    # parser.add_argument('--target', default='france/31TCJ/2017', type=str) denmark/32VNH/2017 austria/33UVP/2017 france/30TXT/2017
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
# ==============================================

    return args

def get_data_loaders(splits, config, balance_source=True):

    strong_aug = transforms.Compose([
            RandomSamplePixels(config.num_pixels),
            RandomSampleTimeSteps(config.seq_length),
            Normalize(),
            ToTensor(),

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
    config = cfg = args
    source_classes = label_utils.get_classes(cfg.source.split('/')[0],
                                             combine_spring_and_winter=cfg.combine_spring_and_winter)
    source_data = PixelSetData(cfg.data_root, cfg.source, source_classes)
    labels, counts = np.unique(source_data.get_labels(), return_counts=True)
    source_classes = [source_classes[i] for i in labels[counts >= 200]]
    print('Using classes:', source_classes)
    cfg.classes = source_classes
    cfg.num_classes = len(source_classes)  # 可以覆盖该参数的默认设置

    # 控制微调样本量
    total_num = len(source_data)  # 获取全量长度
    if args.useall or args.per ==1 :
        use_num = total_num
        print(f"Using all {total_num} samples.")
    elif args.per>1 or args.per <0:
        raise ValueError('Percentage must be between 0 and 1')
    else:
        use_num = round(args.per * total_num)
        print(f"⚠️ Limiting experiment pool to {use_num} random samples (Seed={args.seed}).")
    print(f"(Seed={args.seed}).")

    # Randomly assign parcels to train/val/test
    indices = {config.source: use_num}
    folds = create_train_val_test_folds([config.source], config.num_folds, indices, config.val_ratio,
                                        config.test_ratio)
    splits = folds[0]
    sample_pixels_val = config.sample_pixels_val
    val_loader_dict, test_loader_dict = create_evaluation_loaders(config.source, splits, config, sample_pixels_val)
    source_loader_dict = get_data_loaders(splits, config, config.balance_source)

    num_classes = cfg.num_classes
    args.nclasses = cfg.num_classes
    print("==========>number of classes", num_classes)
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
    # france/31TCJ/2017', type=str) denmark/32VNH/2017 austria/33UVP/2017 france/30TXT/2017
    match args.source:
        case 'france/30TXT/2017':
            source_name = 'FR1'
        case 'france/31TCJ/2017':
            source_name = 'FR2'
        case 'denmark/32VNH/2017':
            source_name = 'DK1'
        case _:
            source_name = 'AT1'


    if finetune:
        model.modelname = f'{source_name}/finetune_R{use_num}_{timestamp}_Seed{args.seed}'
    else:
        model.modelname = f'T_{model.modelname}_R{use_num}_{args.rc_str}_{args.year}_Seed{args.seed}'

    logdir = Path(args.logdir) / model.modelname
    logdir.mkdir(parents=True, exist_ok=True)
    best_model_path = logdir / 'model_best.pth'
    print(f"Logging results to {logdir}")

    criterion = torch.nn.CrossEntropyLoss()
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

            if np.isnan(val_loss) or np.isnan(train_loss):
                print(f"⚠️ Epoch {epoch+1}: Loss detected as NaN! Stopping training to save resources.")
                print("   Possible causes: Learning rate too high, Data normalization issue, or Label mismatch.")
                break # 直接跳出，避免后续错误

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

            # if not_improved_count >= 10:
            #     print("\nValidation performance didn\'t improve for 10 epochs. Training stops.")
            #     break

        if epoch == args.epochs - 1:
            print(f"\n{args.epochs} epochs training finished.")

    # ================= Test Phase =================
    print('Preparing for testing...')
    model.eval()  # 确保模型处于评估模式 (冻结 BN, 禁用 Dropout)

    # 1. 加载最佳模型权重
    if best_model_path.exists():
        print(f'Loading best model from {best_model_path}...')
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            # strict=True 确保架构完全匹配，防止静默失败
            model.load_state_dict(checkpoint['model_state'], strict=True)
            print("✅ Best model loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load best model: {e}")
            if not args.eval:
                raise RuntimeError("Training failed to save a valid model. Cannot proceed with testing.")
            print("⚠️ Proceeding with current model weights (Eval Mode).")
    else:
        if not args.eval:
            raise FileNotFoundError(f"❌ CRITICAL: {best_model_path} not found. Training likely failed (NaN loss).")
        else:
            print("⚠️ No cached best model found. Using currently loaded weights (e.g., pretrained).")

    # 2. 执行测试
    try:
        test_loss, scores = test_epoch_with_adapter(
            model, criterion, test_loader_dict, test_adapter, device, args
        )

        # 格式化输出
        scores_msg = ", ".join(
            [f"{k}={v:.4f}" for k, v in scores.items()
             if k not in ['class_f1', 'confusion_matrix']]
        )
        print(f"\nTest Results:\n{scores_msg}")
        print(f"total_num : {total_num} ; percentage : {args.per*100}%")

        # 保存结果
        scores['epoch'] = 'test'
        scores['testloss'] = test_loss
        scores['total_num'] = total_num
        scores['percentage'] = args.per

        # 提取并单独保存矩阵和 F1
        conf_mat = scores.pop('confusion_matrix', None)
        class_f1 = scores.pop('class_f1', None)

        pd.DataFrame([scores]).set_index("epoch").to_csv(logdir / "testlog.csv")

        if conf_mat is not None:
            np.save(logdir / "test_conf_mat.npy", conf_mat)
        if class_f1 is not None:
            np.save(logdir / "test_class_f1.npy", class_f1)

        print(f"💾 Results saved to {logdir}")

    except Exception as e:
        print(f"❌ Testing phase failed: {e}")
        import traceback
        traceback.print_exc()
        raise
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
    seeds = [111]
    print('seed in', seeds)
    for seed in seeds:
        # args.seed = seed
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
