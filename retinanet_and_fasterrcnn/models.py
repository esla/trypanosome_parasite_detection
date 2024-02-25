import math

# PyTorch and Torchvision imports
import torch
import torchvision


def get_fasterrcnn_resnet101_fpn_v2(args):
    num_classes = args.num_classes

    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models import ResNet101_Weights

    # Create FPN backbone with ResNet architecture
    # New weights with accuracy 80.858%
    backbone = resnet_fpn_backbone(backbone_name='resnet101', weights=ResNet101_Weights.IMAGENET1K_V2,
                                   trainable_layers=5)

    # Create Faster R-CNN model with ResNet50 FPN backbone
    # num_classes is the number of object classes in your dataset + 1 (for background class)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Re-initialize classification and regression heads
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(
        in_features=in_features, out_features=num_classes
    )
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(
        in_features=in_features, out_features=num_classes * 4
    )

    # Initialize classification and regression head weights
    torch.nn.init.normal_(model.roi_heads.box_predictor.cls_score.weight, std=0.01)
    torch.nn.init.normal_(model.roi_heads.box_predictor.bbox_pred.weight, std=0.01)
    torch.nn.init.constant_(model.roi_heads.box_predictor.cls_score.bias, -math.log((1 - 0.01) / 0.01))
    torch.nn.init.constant_(model.roi_heads.box_predictor.bbox_pred.bias, -math.log((1 - 0.01) / 0.01))
    return model

def get_fasterrcnn_resnet50_fpn_v2(args):
    """
    This class only loads a pretrained backbone and initializes the classification and regression heads for Faster R-CNN
    Args:
        args:
    Returns:
    """

    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models import ResNet50_Weights

    num_classes = args.num_classes
    backbone_weights = args.weights_backbone

    # Create FPN backbone with ResNet architecture
    # New weights with accuracy 80.858%
    backbone = resnet_fpn_backbone('resnet50', weights=backbone_weights, trainable_layers=5)

    # Create Faster R-CNN model with ResNet50 FPN backbone
    # num_classes is the number of object classes in your dataset + 1 (for background class)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Re-initialize classification and regression heads
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(
        in_features=in_features, out_features=num_classes
    )
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(
        in_features=in_features, out_features=num_classes * 4
    )

    # Initialize classification and regression head weights
    torch.nn.init.normal_(model.roi_heads.box_predictor.cls_score.weight, std=0.01)
    torch.nn.init.normal_(model.roi_heads.box_predictor.bbox_pred.weight, std=0.01)
    torch.nn.init.constant_(model.roi_heads.box_predictor.cls_score.bias, -math.log((1 - 0.01) / 0.01))
    torch.nn.init.constant_(model.roi_heads.box_predictor.bbox_pred.bias, -math.log((1 - 0.01) / 0.01))
    return model


def get_fasterrcnn_resnet50_fpn_v2_pretrained(args):
    """
    This class loads a pretrained RetinaNet model and only initializes the classification and regression heads
    and number of classes.
    Args:
        args:

    Returns:

    """
    num_classes = args.num_classes
    weights = args.weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    # Load a pretrained Faster RCNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one (adjust for the new number of classes)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_retinanet_resnet50_fpn_v2(args):
    import math
    from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
    weights = args.weights
    num_classes = args.num_classes
    model = retinanet_resnet50_fpn_v2(weights=weights)

    # replace classification layer
    out_channels = model.head.classification_head.conv[0].out_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = num_classes

    cls_logits = torch.nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
    torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
    torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per PyTorch code
    # assign cls head to model
    model.head.classification_head.cls_logits = cls_logits

    return model


def get_retinanet_resnet50_fpn_v2_pretrained(args):
    import torchvision.models.detection
    from torchvision.models.detection.retinanet import RetinaNetHead
    from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
    def modify_retinanet_for_new_classes(model, num_classes):
        # Assuming `model` is a pre-trained RetinaNet model

        # Get the number of input channels and anchor boxes from the existing head
        in_channels = model.head.classification_head.conv[0][0].in_channels
        num_anchors = model.head.classification_head.num_anchors

        # Create a new RetinaNet head with the desired number of classes
        # Note: This retains the structure but requires reinitialization of some parts
        new_head = RetinaNetHead(in_channels, num_anchors, num_classes)

        # Replace the model's head with the new head
        model.head = new_head

        return model

    num_classes = args.num_classes
    weights = args.weights
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=weights)
    model = modify_retinanet_for_new_classes(model, num_classes)
    return model

def get_model(args):
    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
    if args.data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    try:
        if args.model == "fasterrcnn_resnet50_fpn_v2":
            model = get_fasterrcnn_resnet50_fpn_v2(args)
        elif args.model == "retinanet_resnet50_fpn_v2":
            model = get_retinanet_resnet50_fpn_v2(args)
        elif args.model == "fasterrcnn_resnet50_fpn_v2_pretrained":
            model = get_fasterrcnn_resnet50_fpn_v2_pretrained(args)
        elif args.model == "retinanet_resnet50_fpn_v2_pretrained":
            model = get_retinanet_resnet50_fpn_v2_pretrained(args)
        elif args.model == "fasterrcnn_resnet101_fpn_v2":
            model = get_fasterrcnn_resnet101_fpn_v2(args)
    except Exception as e:
        print("Error loading them model: ", e)
    return model