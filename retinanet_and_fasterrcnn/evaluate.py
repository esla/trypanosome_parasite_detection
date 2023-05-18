import os
import json
import torch.utils.data
import detection.utils as utils
from detection.engine import evaluate_test_only

from helper_functions import get_coco_dataset, get_test_args_parser
from helper_functions import get_test_transform
from models import *


def main(args):

    # Fixme: Assert that all needed parameters are given
    # Ensure all necessary parameters are given
    assert args.output_dir, "Provide the output directory"
    assert args.test_images_dir, "Provide the root directory to the test images"
    assert args.test_json_file, "Provide the path to the test JSON file"
    assert args.device, "Provide the device"
    assert args.workers, "Provide the number of workers"
    assert args.result_coco_file, "Provide the path to the COCO results file"

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    # Data loading code
    print("Loading data ...")
    # create the dataset instances
    test_dataset = get_coco_dataset(args.test_images_dir, args.test_json_file, get_test_transform())

    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    else:
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=1, sampler=test_sampler,
                                                   num_workers=args.workers, collate_fn=utils.collate_fn)

    print("Creating model ...")
    model = get_model(args)

    print(f"Loading checkpoint {args.checkpoint} ...")
    checkpoint = torch.load(args.checkpoint)["model"]
    model.load_state_dict(checkpoint)

    model.to(device)
    if args.distributed:
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    torch.backends.cudnn.deterministic = True
    dest_json_file_path = args.result_coco_file
    root_dir = os.path.dirname(dest_json_file_path)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    print(f"Evaluating the model on the test set ...")
    coco_evaluator, results_json = evaluate_test_only(model, data_loader_test, device=device)

    print(f"Saving the results to {dest_json_file_path} ...")
    with open(dest_json_file_path, 'w') as fd:
        json.dump(results_json, fd)
    return


if __name__ == "__main__":
    args = get_test_args_parser().parse_args()
    main(args)
