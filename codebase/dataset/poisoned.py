
import torch
import numpy as np
import torchvision.transforms as transforms


class PoisonedDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            clean_dset,                          # the clean dataset
            poison_rate: float,                  # the rate of poisoned data
            poison_target: int,                  # the target label of poisoned data
            trigger: np.array,                   # the trigger with shape = [H, W, 3] and type = np.uint8
            trigger_alpha: float=None,           # can make the trigger semitransparent.
            trigger_loc: list=None,              # the 2D location to stamp this trigger
            trigger_mask: np.array=None,         # trigger mask with shape [H, W]. (shapes of trigger and input are the same if mask != None)
            poison_seed: int=None,               # seed to generate indices of poisoned data
            only_extract_poisoned: bool=False,   # only return poisoned data that with labels different from the target.

    ) -> None:
        super().__init__()

        self.to_tensor = transforms.ToTensor()
        self.clean_dset = clean_dset

        # only data with the index in all_idx will be returned.
        # by default, all_idx includes all the indices
        self.all_idx = list(range(len(clean_dset)))

        assert trigger.dtype == np.uint8
        self.trigger = trigger
        self.poison_target = poison_target
        self.trigger_loc = trigger_loc
        self.trigger_alpha = trigger_alpha

        self.trigger_mask = trigger_mask
        if trigger_mask is not None:
            assert trigger_loc is None and trigger_alpha is None, "Do not use location and alpha if mask is provided."

        # poison_idx_lst is a list to store indices of poisoned data.
        # it must be initialized as empty here since __getitem__ will access it to get valid data to poison.
        self.poison_idx_lst = []

        # get indices of data that have labels different from the target
        # poison_idx_lst is empty now so no trigger is stamped.
        valid_idx_lst = []
        rand_idx_lst = np.random.RandomState(poison_seed).permutation(len(self))
        for rand_idx in rand_idx_lst:
            x, y = self[rand_idx]
            if y != poison_target:
                valid_idx_lst.append(rand_idx)

            if len(valid_idx_lst) >= int(poison_rate * len(self)):
                break

        # save the indices of data that we will stamp the trigger to
        self.poison_idx_lst = valid_idx_lst

        if only_extract_poisoned is True:
            # only return index of poisoned data
            # this is useful to calculate attack success rates.
            self.all_idx = self.poison_idx_lst

        return

    def __len__(self):
        return len(self.all_idx)

    def __getitem__(self, index):
        # if only_extract_poisoned is False, index does not change
        # otherwise, index will be mapped to an index of poisoned data
        index = self.all_idx[index]

        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        x, y = self.clean_dset.data[index], self.clean_dset.targets[index]

        if index in self.poison_idx_lst:
            # let's stamp the trigger to this data and change its label to the target label
            if self.trigger_loc is not None:
                trigger_size = self.trigger.shape[:2]
                trigger_area = x[
                               self.trigger_loc[0]:self.trigger_loc[0] + trigger_size[0],
                               self.trigger_loc[1]:self.trigger_loc[1] + trigger_size[1],
                               :
                               ]

                modified_img = trigger_area * (1 - self.trigger_alpha) + self.trigger * self.trigger_alpha
                modified_img = modified_img.astype(np.uint8)
                trigger_area[:,:,:] = modified_img

            else:
                # use mask to locate the trigger area.
                trigger_mask = self.trigger_mask.reshape(*self.trigger_mask.shape, 1)
                x = x * (1 - trigger_mask) + self.trigger * trigger_mask
                x = x.astype(np.uint8)

            y = self.poison_target

        if self.clean_dset.transform is not None:
            x = self.clean_dset.transform(x)

        if self.clean_dset.target_transform is not None:
            y = self.clean_dset.target_transform(y)

        return x, y




