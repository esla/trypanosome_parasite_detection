r"""PyTorch Detection Training from Torchvision, adapted by Esla.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
import os
import time
import json
import numpy as np

# import detection.presets as presets
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import detection.utils as utils
# from detection.coco_utils import get_coco_mod, get_coco_kp
from detection.engine import evaluate_one_epoch, train_one_epoch, evaluate, evaluate_test_only
from detection.group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler

from torch.optim.lr_scheduler import CyclicLR
# from models import get_retinanet_resnet50_fpn_v2, get_retinanet_mobilenet_v2
from models import get_fasterrcnn_resnet50_fpn_v2, get_fasterrcnn_resnet101_fpn_v2
from tensorboard_utils import init_tensorboard_logger

# From refactoring
from helper_functions import get_coco_dataset, get_train_args_parser, copypaste_collate_fn
from helper_functions import get_train_transform, get_train_transform_from_weights, get_test_transform
from models import *


def main(args):
    # ToDo: Add assert messages to eliminate some anticipated errors, such as difference in the number of classes
    # between the dataset and the entered args.num_classes

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    # ### start: for tensorboard ###
    tensorboard_logger = None
    if utils.is_main_process() and not args.test_only:
        # passing argparse config with hyperparameters

        tensorboard_logger = init_tensorboard_logger(args)
    # ### end: for tensorboard ###

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    # Data loading code
    print("Loading the training and validation data ...")
    train_transform = get_train_transform_from_weights(args) if args.weights else get_train_transform(args)
    train_dataset = get_coco_dataset(args.train_images_dir, args.train_json_file, train_transform)
    val_dataset = get_coco_dataset(args.val_images_dir, args.val_json_file, get_test_transform())

    print("Creating the data loaders ...")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    train_collate_fn = utils.collate_fn
    if args.use_copypaste:
        if args.data_augmentation != "lsj":
            raise RuntimeError("SimpleCopyPaste algorithm currently only supports the 'lsj' data augmentation policies")

        train_collate_fn = copypaste_collate_fn

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=train_collate_fn
    )
    data_loader_val = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, sampler=val_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    print("Creating the model ...")
    model = get_model(args)
    model.to(device)
    if args.distributed and args.sync_bn:
        if args.sync_bn:
            print("Using Synchronized BatchNorm")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    if args.norm_weight_decay is None:
        parameters = [p for p in model.parameters() if p.requires_grad]
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{"params": p, "weight_decay": w} for p, w in zip(param_groups, wd_groups) if p]

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps,
                                                            gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == "cyclelr":
        # Fixme: The following parameters are hard-coded for now. We need to make them configurable.
        lr_scheduler = CyclicLR(optimizer,
                                base_lr=1e-6,  # Minimum learning rate
                                max_lr=args.lr,  # Maximum learning rate
                                step_size_up=20,  # Number of training iterations in the increasing half of a cycle
                                mode="triangular2")
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    print("Start training")
    start_time = time.time()

    # Fixme: Now, all best epochs are save. This is memory inefficient. Others may need to save only the last
    #  best epoch. Add a config parameter that allows the user to save only the last best model (one model only).
    # ### start: save the best epoch ###
    min_loss = -math.inf  # chose this value arbitrarily after observing the training logs
    # ### end: save the best epoch ###

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        print(" -------------- Model Training --------------------")
        train_info = train_one_epoch(model, optimizer, data_loader_train, device, epoch, args.print_freq, scaler)

        train_info = {key: str(elem) for key, elem in train_info.meters.items()}

        learning_rate = torch.tensor(float(train_info["lr"]))
        train_info.pop("lr")
        train_info = {"train/" + key: torch.tensor(float(elem.split(" ")[1].strip("()")))
                      for key, elem in
                      train_info.items()}  # casting to tensor.float because of tensorboard's add_scalar
        train_info["train/lr"] = learning_rate

        lr_scheduler.step()  # update lr_scheduler only after an epoch

        # evaluate after every epoch
        print(" -------------- Model evaluation (losses) --------------------")
        eval_info = evaluate_one_epoch(model, optimizer, data_loader_val, device, epoch, args.print_freq, scaler)
        eval_info = {key: str(elem) for key, elem in eval_info.meters.items()}
        learning_rate = torch.tensor(float(eval_info["lr"]))
        eval_info.pop("lr")
        eval_info = {"eval/" + key: torch.tensor(float(str(elem).split(" ")[1].strip("()")))
                     for key, elem in eval_info.items()}  # casting to tensor.float because of tensorboard's add_scalar
        eval_info["eval/lr"] = learning_rate

        # Fixme: Consider moving this to the evaluation loop. Logging this info is useful but slows training as it is.
        print(" --------------- Model Evaluation (mAP) -----------------------")
        coco_results = evaluate(model, data_loader_val, device=device)
        eval_info["eval/mAP50"] = coco_results.coco_eval['bbox'].stats[1]

        log_info = dict(train_info.items() | eval_info.items())

        # ### start: for tensorboard ###
        if utils.is_main_process():
            if utils.is_main_process():
                # track losses
                tensorboard_logger.add_scalar_dict(
                    # passing the dictionary of losses (pairs - loss_key: loss_value)
                    log_info,
                    # passing the global step (number of iterations)
                    global_step=epoch,
                    # adding the tag to combine the plots in a subgroup
                    tag="losses"
                )
            # ### end: for tensorboard ###

        # ### start: save the best epoch ###
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()

            curr_loss = eval_info["eval/loss"]

            if curr_loss <= min_loss:
                print("*" * 40)
                print(f"Saving ... The last best was {min_loss}")
                print("*" * 40)
                min_loss = curr_loss

                utils.save_on_master(
                    {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'args': args
                    },
                    os.path.join(args.output_dir, 'best_model_{}.pth'.format(epoch)))

        # ### end: save the best epoch ###

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_train_args_parser().parse_args()
    main(args)
