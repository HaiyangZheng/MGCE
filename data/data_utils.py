import numpy as np
from torch.utils.data import Dataset
import copy

def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices

class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

        ## Merge the data and targets from both datasets for DCCL 
        _labelled_dataset = copy.deepcopy(labelled_dataset)
        _unlabelled_dataset = copy.deepcopy(unlabelled_dataset)
        if unlabelled_dataset is not None:
            if hasattr(_labelled_dataset, 'data'):
                if isinstance(_labelled_dataset.data, list):
                    self.data = _labelled_dataset.data + _unlabelled_dataset.data
                else:
                    self.data = np.concatenate((_labelled_dataset.data, _unlabelled_dataset.data), axis=0)
            elif hasattr(_labelled_dataset, 'samples'):
                 self.data = _labelled_dataset.samples + _unlabelled_dataset.samples
            else:
                assert False, f'Unsuport {labelled_dataset}'

            if hasattr(_labelled_dataset, 'targets') or hasattr(_labelled_dataset, 'target'):
                lt = _labelled_dataset.targets if hasattr(_labelled_dataset, 'targets') else _labelled_dataset.target
                ut = _unlabelled_dataset.targets if hasattr(_unlabelled_dataset, 'targets') else _unlabelled_dataset.target
                if isinstance(lt, list):
                    self.targets = lt + ut
                else:
                    self.targets = np.concatenate((lt, ut), axis=0)
            else:
                assert False, f'Unsuport {labelled_dataset}'

            if hasattr(labelled_dataset, 'uq_idxs'):
                self.uq_idxs = labelled_dataset.uq_idxs.tolist() + unlabelled_dataset.uq_idxs.tolist()
            else:
                assert False, f'Unsuport {labelled_dataset}'

    def __getitem__(self, item):

        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1

        else:

            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0


        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)
