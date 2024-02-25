import torchvision
from torchvision.transforms import InterpolationMode
from detection import presets
from detection.transforms import SimpleCopyPaste
from detection.coco_utils import ConvertCocoPolysToMask, CocoDetection, _coco_remove_images_without_annotations
from detection import utils
import detection.transforms as T


def get_train_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--train-images-dir", type=str,
                        help="root directory of the training images path")
    parser.add_argument("--val-images-dir", type=str,
                        help="root directory of the validation images path")
    parser.add_argument("--test-images-dir", type=str,
                        help="root directory of the test images path")
    parser.add_argument("--train-json-file", type=str,
                        help="coco json file containing the training dataset")
    parser.add_argument("--val-json-file", type=str,
                        help="coco json file containing the validation dataset")
    parser.add_argument("--test-json-file", type=str,
                        help="coco json file containing the test dataset")
    parser.add_argument("--num-classes", default=2, type=int,
                        help="number of classes in the dataset")
    parser.add_argument("--dataset", default="coco", type=str,
                        help="dataset name")
    parser.add_argument("--partition", default="val", type=str,
                        help="dataset partition (val/test)")
    parser.add_argument("--model", default="fasterrcnn_resnet50_fpn_v2", type=str,
                        help="model name")
    parser.add_argument("--device", default="cuda", type=str,
                        help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=2, type=int,
                        help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--epochs", default=60, type=int, metavar="N",
                        help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                        help="number of data loading workers (default: 4)")
    parser.add_argument("--opt", default="sgd", type=str,
                        help="optimizer")
    parser.add_argument("--lr", default=0.02, type=float,
                        help="initial learning rate, 0.02 is the default value for training on 8 "
                             "gpus and 2 images_per_gpu")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M",
                        help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W",
                        help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--norm-weight-decay", default=None, type=float,
                        help="weight decay for Normalization layers (default: None, same value as --wd)")
    parser.add_argument("--lr-scheduler", default="multisteplr", type=str,
                        help="name of lr scheduler (default: multisteplr)")
    parser.add_argument("--lr-step-size", default=8, type=int,
                        help="decrease lr every step-size epochs (multisteplr scheduler only)")
    parser.add_argument("--lr-steps", default=[16, 22], nargs="+", type=int,
                        help="decrease lr every step-size epochs (multisteplr scheduler only)")
    parser.add_argument("--lr-gamma", default=0.1, type=float,
                        help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)")
    parser.add_argument("--print-freq", default=50, type=int,
                        help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str,
                        help="path to save outputs")
    parser.add_argument("--checkpoint", default="", type=str,
                        help="path of checkpoint for evaluation")
    parser.add_argument("--resume", default="", type=str,
                        help="path of checkpoint to resume from")
    parser.add_argument("--start_epoch", default=0, type=int,
                        help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float,
                        help="rpn score threshold for faster-rcnn")
    parser.add_argument("--trainable-backbone-layers", default=None, type=int,
                        help="number of trainable layers of backbone")
    parser.add_argument("--data-augmentation", default="hflip", type=str,
                        help="data augmentation policy (default: hflip)")
    parser.add_argument("--sync-bn", dest="sync_bn",
                        help="Use sync batch norm", action="store_true")
    parser.add_argument("--test-only", dest="test_only",
                        help="Only test the model", action="store_true")
    parser.add_argument("--tensorboard-rootdir", type=str,
                        help="root dir to save the tensorboard log outputs")
    parser.add_argument("--use-deterministic-algorithms", action="store_true",
                        help="Forces the use of deterministic algorithms only.")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int,
                        help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str,
                        help="url used to set up distributed training")
    parser.add_argument("--weights", default=None, type=str,
                        help="the weights enum name to load for the object detection model")
    parser.add_argument("--weights-backbone", default=None, type=str,
                        help="the backbone weights enum name to load")
    parser.add_argument("--result-coco-file", default=None, type=str,
                        help="coco file name to save the predictions to")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true",
                        help="Use torch.cuda.amp for mixed precision training")

    # Use CopyPaste augmentation training parameter
    parser.add_argument("--use-copypaste", action="store_true",
                        help="Use CopyPaste data augmentation. Works only with data-augmentation='lsj'.")
    return parser


def get_test_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Object Detection Evaluation", add_help=add_help)
    parser.add_argument("--test-images-dir", type=str,
                        help="root directory of the test images path")
    parser.add_argument("--test-json-file", type=str,
                        help="coco json file containing the test dataset")
    parser.add_argument("--num-classes", default=2, type=int,
                        help="number of classes in the dataset")
    parser.add_argument("--dataset", default="coco", type=str,
                        help="dataset name")
    parser.add_argument("--model", default="fasterrcnn_resnet50_fpn_v2", type=str,
                        help="model name")
    parser.add_argument("--device", default="cuda", type=str,
                        help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=2, type=int,
                        help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                        help="number of data loading workers (default: 4)")
    parser.add_argument("--print-freq", default=50, type=int,
                        help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str,
                        help="path to save outputs")
    parser.add_argument("--checkpoint", default="", type=str,
                        help="path of checkpoint for evaluation")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float,
                        help="rpn score threshold for faster-rcnn")
    parser.add_argument("--trainable-backbone-layers", default=None, type=int,
                        help="number of trainable layers of backbone")
    parser.add_argument("--data-augmentation", default="hflip", type=str,
                        help="data augmentation policy (default: hflip)")
    parser.add_argument("--sync-bn", dest="sync_bn",
                        help="Use sync batch norm", action="store_true")
    parser.add_argument("--use-deterministic-algorithms", action="store_true",
                        help="Forces the use of deterministic algorithms only.")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int,
                        help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str,
                        help="url used to set up distributed training")
    parser.add_argument("--result-coco-file", default=None, type=str,
                        help="coco file name to save the predictions to")
    parser.add_argument("--amp", action="store_true",
                        help="Use torch.cuda.amp for mixed precision training")
    return parser


def get_coco_dataset(img_root, ann_file, transforms, mode="instances"):
    t = [ConvertCocoPolysToMask()]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    dataset = CocoDetection(img_root, ann_file, transforms=transforms)

    if mode == "train":
        dataset = _coco_remove_images_without_annotations(dataset)

    return dataset


def copypaste_collate_fn(batch):
    copypaste = SimpleCopyPaste(blending=True, resize_interpolation=InterpolationMode.BILINEAR)
    return copypaste(*utils.collate_fn(batch))


def get_train_transform(args):
        return presets.DetectionPresetTrain(data_augmentation=args.data_augmentation)


def get_test_transform():
    return presets.DetectionPresetEval()


def get_train_transform_from_weights(args):
    weights = torchvision.models.get_weight(args.weights)
    trans = weights.transforms()
    return lambda img, target: (trans(img), target)


