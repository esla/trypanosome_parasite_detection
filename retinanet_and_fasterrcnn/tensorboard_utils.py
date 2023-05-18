import os
import tensorboardX


class SummaryWriter(tensorboardX.SummaryWriter):
    """
    SummaryWriter class inherits from tensorboardX.SummaryWriter.

    Attributes:
    -----------
    args : dict
        Argument parser dictionary, initially None.
    total_epochs : int
        Total number of epochs, initially 0.
    global_iter : int
        Total number of iterations, initially 0.
    INSTANCE_CATEGORY_NAMES : list
        List of instance category names.
    """

    INSTANCE_CATEGORY_NAMES = ["__background__", "parasite"]

    def __init__(self, log_dir=None, comment="", **kwargs):
        """
        Initialize SummaryWriter object.

        Parameters:
        -----------
        log_dir : str
            Directory for the log files.
        comment : str
            Comment for the log files.
        kwargs : dict
            Other keyword arguments.
        """
        super().__init__(log_dir, comment, **kwargs)
        self.args = None  # Will be replaced to argparse dictionary
        self.total_epochs = 0  # Initialize the total number of epochs
        self.global_iter = 0  # Initialize the total number of iterations

    def add_scalar_dict(self, dictionary, global_step, tag=None):
        """
        Add scalar values from a dictionary to the tensorboard.

        Parameters:
        -----------
        dictionary : dict
            Dictionary of scalar values.
        global_step : int
            Global step value.
        tag : str
            Tag for the dictionary.
        """
        for name, val in dictionary.items():
            if tag is not None:
                name = os.path.join(tag, name)
            self.add_scalar(name, val.item(), global_step)


def init_tensorboard_logger(args):
    model_result_foldername = args.output_dir.split("/")[-1]
    tensorboard_dir = os.path.join(args.tensorboard_rootdir, model_result_foldername)
    tensorboard_logger = SummaryWriter(log_dir=tensorboard_dir)
    return tensorboard_logger