'''
Since we cannot provide the datasets, you have to implement your own Dataset class in src/core/utils/dataset which should
return two rgb images with shape [3, height, width] containing the same person and background, but in different poses.
'''

from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Implement your dataset class here!"""

    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
