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
        self.warn_count = 0
        self.invalid_sample_warn_count = 0

    def __call__(self, batch_dict):
        pixels = batch_dict['pixels']
        positions = batch_dict['positions']
        valid_pixels = batch_dict['valid_pixels']
        pixel_labels = batch_dict['pixel_labels']

        B, T, C, N = pixels.shape

        # 1. 展平
        data_flat = pixels.permute(0, 3, 1, 2).reshape(-1, T, C)
        doy_flat = positions.unsqueeze(1).expand(-1, N, -1).reshape(-1, T)
        valid_flat = valid_pixels.permute(0, 2, 1).reshape(-1, T).bool()
        mask_flat = ~valid_flat
        y_flat = pixel_labels.reshape(-1).long()

        # ==========================================
        # 👇 【关键修复 1】防止“全无效样本”导致模型内除零 (Div-by-Zero)
        # 如果某样本所有时间步都无效 (valid_flat 全 False)，模型计算 weight/sum(weight) 时会除以 0 -> NaN
        # 解决方案：强制将这些样本的第一个时间步标记为有效。
        # ==========================================
        has_valid_time = valid_flat.any(dim=1)
        all_invalid_mask = ~has_valid_time  # 布尔掩码：True 表示该样本全无效

        if all_invalid_mask.any():
            count = all_invalid_mask.sum().item()
            if self.invalid_sample_warn_count < 5:
                print(f"⚠️ WARNING: Detected {count} samples with ALL time steps invalid. "
                      f"Forcing first time step to VALID to prevent Div-by-Zero NaN.")
                self.invalid_sample_warn_count += 1

            # 强制修正：将全无效样本的第 0 个时间步设为 True (有效)
            valid_flat[all_invalid_mask, 0] = True
            mask_flat[all_invalid_mask, 0] = False

            # 重新计算 has_valid_time (现在它们都是 True 了)
            has_valid_time = valid_flat.any(dim=1)

        # ==========================================
        # 👇 【关键修复 2】处理输入数据中的 NaN/Inf
        # ==========================================
        has_nan_inf = torch.isnan(data_flat).any(dim=(1, 2)) | torch.isinf(data_flat).any(dim=(1, 2))

        if has_nan_inf.any():
            dirty_indices = has_nan_inf
            y_flat = y_flat.float()
            y_flat[dirty_indices] = -100.0
            y_flat = y_flat.long()

            # 将脏数据的有效时间步设为 False (虽然上面已经处理了全无效的情况，但这里是为了逻辑一致)
            # 注意：如果上面强制设为了 True，这里如果数据本身是 NaN，我们依然希望模型忽略它。
            # 但为了防止再次触发全无效逻辑，我们只标记数据，不反转 valid_flat，
            # 因为模型内部应该通过 loss 的 ignore_index 来处理，或者依靠前面的防御性代码。
            # 最安全的做法：如果数据是 NaN，即使 valid 是 True，模型算出来也是 NaN。
            # 所以这里必须把 valid 设为 False，但这可能再次制造“全无效样本”。
            # 策略：如果数据是 NaN，设为 False。如果因此变成全无效，上面的逻辑会在下一轮 (或模型内) 处理?
            # 不，上面的逻辑已经跑过了。
            # 修正策略：如果数据是 NaN，我们把它设为 0，并保持 valid=True (如果它是唯一的有效步)，
            # 或者设为 valid=False 并接受它可能被上面的逻辑“救活”为第一步有效但数据为 0。

            # 最佳实践：将 NaN 数据置 0，并标记 Label 为 Ignore。
            # 不需要把 valid 设为 False，因为数据已经是 0 了，不会炸，只是没信息量。
            # 但如果原数据是 Inf，置 0 是安全的。

            data_flat[dirty_indices] = 0.0  # 替换脏数据为 0

            if self.warn_count < 5:
                print(f"⚠️ WARNING: Detected {has_nan_inf.sum().item()} samples with NaN/Inf in input data. "
                      f"Replaced with 0 and set Label to Ignore.")
                self.warn_count += 1

        # ==========================================
        # 👇 处理无有效时间步的样本 (Label Ignoring)
        # ==========================================
        # 注意：经过 [关键修复 1] 后，理论上 has_valid_time 应该全为 True。
        # 但为了逻辑健壮性，保留此步骤。
        y_flat = y_flat.float()
        IGNORE_INDEX = -100
        y_flat = torch.where(has_valid_time, y_flat, torch.tensor(IGNORE_INDEX, dtype=y_flat.dtype))
        y_flat = y_flat.long()

        # 再次全局检查 (双重保险)
        if torch.isnan(data_flat).any() or torch.isinf(data_flat).any():
            data_flat = torch.nan_to_num(data_flat, nan=0.0, posinf=0.0, neginf=0.0)

        X_tuple = (
            data_flat,
            mask_flat,
            doy_flat,
            valid_flat.float()
        )

        # 打印一次调试信息
        if self.warn_count < 1:
            print(f"\n--- [INPUT DATA DEBUG] ---")
            print(f"data_flat shape: {data_flat.shape}")
            print(f"data_flat Min: {data_flat.min().item():.4f}, Max: {data_flat.max().item():.4f}")
            print(f"data_flat Mean: {data_flat.mean().item():.4f}, Std: {data_flat.std().item():.4f}")
            print(f"Has NaN/Inf in data_flat: {torch.isnan(data_flat).any() or torch.isinf(data_flat).any()}")
            print(f"mask_flat (True=invalid) sum: {mask_flat.sum().item()}")
            print(f"doy_flat range: [{doy_flat.min().item()}, {doy_flat.max().item()}]")

            # 检查是否还有全无效的样本 (理论上应该没有了)
            final_check = ~valid_flat.any(dim=1)
            if final_check.any():
                print(f"❌ CRITICAL: Still have {final_check.sum().item()} all-invalid samples after fix!")
            else:
                print(f"✅ All samples have at least one valid time step.")

            print(f"--------------------------\n")
            self.warn_count += 1

        if self.device is not None:
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
                continue

            if y.device != device:
                y = y.to(device)

            optimizer.zero_grad()

            if args.use_doy:
                logits = model(X, use_doy=True)
            else:
                logits = model(X)

            # 👇 【关键调试】在第一个 batch 检查数据状态
            if not debug_printed:
                print(f"\n--- [DEBUG] Batch {idx} Status ---")
                print(f"Logits shape: {logits.shape}, Min: {logits.min().item():.4f}, Max: {logits.max().item():.4f}")
                print(f"Logits has NaN/Inf: {torch.isnan(logits).any() or torch.isinf(logits).any()}")

                # 检查标签
                unique_labels = torch.unique(y[y >= 0])  # 只看有效标签
                print(f"Valid Labels unique values: {unique_labels.tolist()}")
                print(f"Max Label Value: {unique_labels.max().item() if len(unique_labels) > 0 else 'None'}")
                print(f"Expected Num Classes: {args.nclasses}")
                if len(unique_labels) > 0 and unique_labels.max().item() >= args.nclasses:
                    print("⚠️ CRITICAL ERROR: Label value >= num_classes! This causes NaN in CrossEntropyLoss.")
                print(f"-------------------------------\n")
                debug_printed = True

            # 👇 【关键防御】如果 Logits 已经是 NaN/Inf，跳过 backward，防止污染优化器状态
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"⚠️ Skipping batch {idx}: Logits contain NaN/Inf.")
                continue

            # 再次确认标签范围 (双重保险)
            if (y >= args.nclasses).any() and (y != -100).any():
                print(f"⚠️ Skipping batch {idx}: Labels out of range detected.")
                # 将越界标签强制设为 -100
                y = y.clone()
                y[y >= args.nclasses] = -100

            out = F.log_softmax(logits, dim=-1)

            # 检查 log_softmax 后是否有 NaN
            if torch.isnan(out).any():
                print(f"⚠️ Skipping batch {idx}: log_softmax output contains NaN.")
                continue

            loss = criterion(out, y)

            # 检查 Loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ Skipping batch {idx}: Loss is NaN/Inf.")
                continue

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

                # if y[y!=6].sum().item() > 0:
                #     print(y)
                if args.use_doy:
                    logits = model(X, use_doy=True)
                else:
                    logits = model(X)

                # 这里可能有问题，train函数同
                out = F.log_softmax(logits, dim=-1)
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
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-5,
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
    parser.add_argument('--num_pixels', default=100, type=int, help='Number of pixels to sample from the input sample')
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
    target_data = PixelSetData(cfg.data_root, cfg.target, source_classes)
    labels, counts = np.unique(source_data.get_labels(), return_counts=True)
    source_classes = [source_classes[i] for i in labels[counts >= 200]]
    print('Using classes:', source_classes)
    cfg.classes = source_classes
    cfg.num_classes = len(source_classes)  # 可以覆盖该参数的默认设置

    # 控制微调样本量
    total_num = len(target_data)  # 获取全量长度
    if args.useall or args.num >= total_num:
        use_num = total_num
        print(f"Using all {total_num} samples.")
    else:
        use_num = args.num
        print(f"⚠️ Limiting experiment pool to {use_num} random samples (Seed={args.seed}).")

    # Randomly assign parcels to train/val/test
    indices = {config.target: use_num}
    folds = create_train_val_test_folds([config.target], config.num_folds, indices, config.val_ratio,
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

    print(f"🔍 Model Config Check:")
    print(f"   args.use_doy: {args.use_doy}")
    if hasattr(model, 'use_doy'):
        print(f"   model.use_doy: {model.use_doy}")
    if hasattr(model, 'pos_encoder'):
        print(f"   model.pos_encoder exists: True")
        # 尝试打印 pos_encoder 的状态
        pe_params = [p for p in model.pos_encoder.parameters()]
        if len(pe_params) > 0:
            print(f"   pos_encoder has parameters: {len(pe_params)}")
            # 检查 PE 权重是否有 NaN
            for name, param in model.pos_encoder.named_parameters():
                if torch.isnan(param).any():
                    print(f"   💣 NaN found in PE parameter: {name}")
            for name, buf in model.pos_encoder.named_buffers():
                if torch.isnan(buf).any():
                    print(f"   💣 NaN found in PE buffer: {name}")
    print("------------------")

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
    print("🛡️ [Stability Fix v2] Freezing ALL BatchNorm and LayerNorm layers in the model...")
    frozen_count = 0
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.LayerNorm)):
            m.eval()
            frozen_count += 1
            # 可选：打印前几个被冻结的层，确认包含 encoder 部分
            if frozen_count <= 5:
                print(f"   - Frozen: {name} ({type(m).__name__})")

    print(f"   Total frozen normalization layers: {frozen_count}")

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

    criterion = torch.nn.CrossEntropyLoss(reduction="mean",ignore_index=-100)
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # 👇 【关键修复 1】针对微调场景优化学习率
    # 如果加载了预训练模型 (finetune=True)，分类头是随机初始化的。
    # 默认 1e-3 可能太大导致第一步就爆炸 (NaN)。建议微调时使用 1e-4 或更小，或者依赖 warmup。
    current_lr = args.learning_rate
    if finetune and args.warmup_epochs == 0:
        # 如果没有 warmup，自动降低初始学习率以防爆炸
        current_lr = args.learning_rate * 0.1
        print(f"⚠️ Finetuning mode detected without warmup. Reducing initial LR to {current_lr} to prevent NaN.")

    optimizer = torch.optim.Adam(parameters, lr=current_lr, weight_decay=args.weight_decay)

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

            if not_improved_count >= 10:
                print("\nValidation performance didn\'t improve for 10 epochs. Training stops.")
                break

        if epoch == args.epochs - 1:
            print(f"\n{args.epochs} epochs training finished.")

    # test
    # ================= Test Phase =================
    print('Preparing for testing...')

    model_loaded_successfully = False

    # 逻辑判断：只有在非 eval 模式 且 文件存在 时才加载 best_model
    if not args.eval and best_model_path.exists():
        print(f'Restoring best model weights from {best_model_path}...')
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            # strict=True 确保所有参数（包括 BN 的 running_mean/var）都匹配并加载
            model.load_state_dict(checkpoint['model_state'], strict=True)
            if 'criterion' in checkpoint:
                criterion = checkpoint['criterion']
            model_loaded_successfully = True
            print("Best model loaded successfully.")
        except Exception as e:
            print(f"Failed to load best model: {e}. Will attempt to reload pretrained weights.")
            model_loaded_successfully = False

    elif args.eval and best_model_path.exists():
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state'], strict=True)
            print("Loaded cached best model for evaluation.")
        except:
            print("Using currently loaded (pretrained) weights.")

    else:
        # === 关键修复区域：训练失败后的恢复逻辑 ===
        if not args.eval:
            print(f"⚠️ CRITICAL: {best_model_path} does not exist.")
            print("   Training failed (NaN loss). Current model weights/statistics are likely CORRUPTED.")

            if finetune and args.pretrained:
                print("   ⚡ FORCE RELOADING original PRETRAINED weights to clean corrupted BN statistics...")

                try:
                    path = Path(args.pretrained).absolute().relative_to(Path(__file__).absolute().parent)
                    checkpoint = torch.load(path, map_location=device)  # 加载整个 checkpoint
                    pretrain_state = checkpoint['model_state']

                    # 1. 获取当前模型状态字典
                    model_dict = model.state_dict()

                    # 2. 过滤预训练权重 (只保留 encoder_q 部分，去掉 decoder/classifier)
                    state_dict = {}
                    for k, v in pretrain_state.items():
                        if k.startswith('encoder_q'):
                            if 'decoder' not in k and 'classification' not in k and 'position_enc.pe' not in k:
                                new_key = k[len("encoder_q."):]
                                if new_key in model_dict:
                                    state_dict[new_key] = v
                                # 注意：这里可能漏掉了 BN 的 running_mean/var 如果它们名字不匹配
                                # 但通常 MoCo 的 key 是直接对应的

                    # 3. 【关键】保留当前模型中未被预训练权重覆盖的部分 (如随机初始化的 classifier)
                    # 但对于 BN 层，我们希望用预训练的统计量覆盖脏统计量。
                    # 如果 pretrain_state 里有 BN 的 buffer，上面的循环应该已经包含了。
                    # 如果没包含（比如 key 名字变了），我们需要手动处理。

                    # 4. 执行加载 (strict=False 允许 classifier 不匹配，但 BN 必须匹配)
                    # 如果 strict=True 报错，说明 key 不匹配，我们需要更精细的处理
                    try:
                        model.load_state_dict(state_dict, strict=False)
                        print("   ✅ Pretrained encoder weights loaded.")

                        # 5. 【双重保险】手动检查并重置 BN 层
                        # 如果加载后还有 NaN，说明某些 BN 层没被覆盖。
                        # 这里我们强制将所有 BN 层设为 eval 模式，防止它们使用可能残留的脏统计量
                        # 或者，如果可能，重新初始化 BN 层 (但这会丢失预训练统计量)
                        # 最安全的做法：既然我们是微调，且预训练模型是好的，
                        # 我们应该确保所有 BN 层的 running_var 不是 NaN。

                        nan_bn_count = 0
                        for name, m in model.named_modules():
                            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                                if torch.isnan(m.running_var).any() or torch.isinf(m.running_var).any():
                                    nan_bn_count += 1
                                    # 如果发现 NaN，尝试从预训练字典里再找一次，或者重置为 1
                                    # 这里简单粗暴：如果 var 是 NaN，设为 1.0 (相当于不缩放)
                                    m.running_var.fill_(1.0)
                                    m.running_mean.fill_(0.0)
                                    print(f"   🛠️ Fixed NaN in BN layer: {name}")

                        if nan_bn_count > 0:
                            print(f"   ⚠️ Detected and fixed {nan_bn_count} corrupted BN layers.")
                        else:
                            print("   ✅ All BN statistics appear clean.")

                    except Exception as load_err:
                        print(f"   ❌ Failed to load state dict: {load_err}")
                        print("   Falling back to using the model as-is (high risk of NaN).")

                except Exception as e:
                    print(f"   ❌ Error reloading pretrained checkpoint: {e}")
            else:
                print("   Proceeding with current model state (likely corrupted).")
        else:
            print("Eval mode: No cached model found. Using provided pretrained weights.")

    # === 执行测试 ===
    # 确保模型处于评估模式 (这会禁用 BN 的统计量更新，并使用 running_mean/var)
    model.eval()

    # 执行测试
    try:
        test_loss, scores = test_epoch_with_adapter(model, criterion, test_loader_dict, test_adapter, device, args)

        scores_msg = ", ".join(
            [f"{k}={v:.4f}" for (k, v) in scores.items() if k not in ['class_f1', 'confusion_matrix']])
        print(f"Test results: \n\n {scores_msg}")

        scores['epoch'] = 'test'
        scores['testloss'] = test_loss

        conf_mat = scores.pop('confusion_matrix', None)
        class_f1 = scores.pop('class_f1', None)

        log_df = pd.DataFrame([scores]).set_index("epoch")
        log_df.to_csv(logdir / f"testlog.csv")

        if conf_mat is not None:
            np.save(logdir / f"test_conf_mat.npy", conf_mat)
        if class_f1 is not None:
            np.save(logdir / f"test_class_f1.npy", class_f1)

    except Exception as e:
        print(f"❌ Testing phase failed: {e}")
        import traceback
        traceback.print_exc()

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
