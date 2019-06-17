"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
from teupy import SegmentationDataset, RemoteHandler, LabelParser, Rotation, RandomCrop, TransformToTensor
import os
import random
import torchvision.transforms as transforms
# from data.image_folder import make_dataset
# from PIL import Image


class LymphoDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--lp_nameA', type=str, default='Neutrophiler (reif)', help='label parser group')
        parser.add_argument('--lp_nameB', type=str, default='Neutrophiler (unreif)', help='label parser group')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)

        remote_handler = RemoteHandler('Minimal Working Example', force_work='scratch', user='temirlan')
        label_parserA = LabelParser(groups=[self.opt.lp_nameA])
        label_parserB = LabelParser(groups=[self.opt.lp_nameB])

        self.transform = [Rotation(), RandomCrop(offset=-1, patch_size=256), TransformToTensor(normalise=True)]
        # self.transform2 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.segm_datasetA = SegmentationDataset(size_minibatches=1,
							num_subsets=6,
							patchsize=256,
							label_parser=label_parserA,
							transform=self.transform,
							return_parameters=['image', 'segmap_NCWH'],
							remote_handler=remote_handler)
        self.segm_datasetA.load_sample_list(load_from=os.path.join('slide002_6x1000x400x200_centred.npy'))
        self.segm_datasetA.keep_in_memory()
        
        self.segm_datasetB = SegmentationDataset(size_minibatches=1,
							num_subsets=6,
							patchsize=256,
							label_parser=label_parserB,
							transform=self.transform,
							return_parameters=['image', 'segmap_NCWH'],
							remote_handler=remote_handler)
        self.segm_datasetB.load_sample_list(load_from=os.path.join('slide002_6x1000x400x200_centred.npy'))
        self.segm_datasetB.keep_in_memory()

        if self.opt.isTrain:
            self.segm_datasetA.subset = [i for i in range(self.segm_datasetA.num_subsets) if i != self.opt.test_subset]
            self.segm_datasetB.subset = [i for i in range(self.segm_datasetA.num_subsets) if i != self.opt.test_subset]
        else:
            self.segm_datasetA.subset = self.opt.test_subset
            self.segm_datasetB.subset = self.opt.test_subset

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        # PROBLEM: no randomness
        # IDEA: segm_dataset without transforms, add transforms after extracting images and segmap
        data_A = self.segm_datasetA[index % len(self.segm_datasetA)]['image'].squeeze(0)    # needs to be a tensor
        if self.opt.serial_batches:   # make sure index is within the range
            index_B = index % len(self.segm_datasetB)
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, len(self.segm_datasetB) - 1)
        data_B = self.segm_datasetB[index_B]['image'].squeeze(0)

        return {'A': data_A, 'B': data_B, 'A_paths': [], 'B_paths': []}

    def __len__(self):
        """Return the total number of images."""
        return max(len(self.segm_datasetA), len(self.segm_datasetB))
