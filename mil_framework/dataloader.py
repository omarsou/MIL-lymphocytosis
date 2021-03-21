from torchtext import data
from torchtext.data import Dataset, Iterator, Field
import torch
from typing import Tuple
from mil_framework.utils import load_dataset_file
import os


class LymphoDataset(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
            self,
            path: str,
            include: list,
            fields: Tuple[Field, Field, Field, Field, Field],
            **kwargs
    ):

        path = [path]

        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("id", fields[0]),
                ("age", fields[1]),
                ("concentration", fields[2]),
                ("features", fields[3]),
                ("target", fields[4])
            ]

        samples = {}
        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for aut_id in tmp:
                if include:
                    if aut_id not in include:
                        continue
                s = tmp[aut_id]
                samples[aut_id] = {
                    "id": aut_id,
                    "age": s["age"],
                    "concentration": s["concentration"],
                    "features": s["features"],
                    "target": s["label"],
                }

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["id"],
                        # This is for numerical stability
                        sample["age"] + 1e-8,
                        sample["concentration"] + 1e-8,
                        sample["features"] + 1e-8,
                        sample["target"],
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)


def load_data(data_cfg, train_names, val_names, test=False):
    data_path = data_cfg.model_dir + data_cfg.data_path
    train_paths = os.path.join(data_path, 'files_efficient.train')
    pad_feature_size = 2560

    def tokenize_features(features):
        ft_list = torch.split(features, 1, dim=0)
        return [ft.squeeze() for ft in ft_list]

    def stack_features(features, something):
        return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)

    id_field = data.RawField()
    id_field.is_target = False

    age_field = data.Field(
        use_vocab=False,
        init_token=None,
        dtype=torch.float32,
        # preprocessing=tokenize_features,
        tokenize=lambda features: features,
        batch_first=True,
        include_lengths=False,
        postprocessing=stack_features
    )

    features_field = data.Field(
        use_vocab=False,
        init_token=None,
        dtype=torch.float32,
        # preprocessing=tokenize_features,
        tokenize=lambda features: features,
        batch_first=True,
        include_lengths=True,
        postprocessing=stack_features,
        pad_token=torch.zeros((pad_feature_size,)),
    )

    concentration_field = data.Field(
        use_vocab=False,
        init_token=None,
        dtype=torch.float32,
        # preprocessing=tokenize_features,
        tokenize=lambda features: features,
        batch_first=True,
        include_lengths=False,
        postprocessing=stack_features
    )

    label_field = data.Field(
        use_vocab=False,
        init_token=None,
        dtype=torch.int32,
        # preprocessing=tokenize_features,
        tokenize=lambda features: features,  # TODO (Cihan): is this necessary?
        batch_first=True,
        postprocessing=stack_features,
    )

    train_data = LymphoDataset(
        path=train_paths,
        include=train_names,
        fields=(id_field, age_field, concentration_field, features_field, label_field)
    )

    dev_data = LymphoDataset(
        path=train_paths,
        include=val_names,
        fields=(id_field, age_field, concentration_field, features_field, label_field)
    )
    if test:
        test_paths = os.path.join(data_path, 'files_efficient.test')
        test_data = LymphoDataset(
            path=test_paths,
            include=None,
            fields=(id_field, age_field, concentration_field, features_field, label_field))
        return train_data, dev_data, test_data
    return train_data, dev_data


def make_data_iter(
        dataset: Dataset,
        batch_size: int,
        train: bool = False,
        shuffle: bool = False,
) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.
    :param dataset: torchtext dataset containing sgn and optionally txt
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
        bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = None

    if train:
        # optionally shuffle and sort during training
        data_iter = data.BucketIterator(
            repeat=False,
            sort=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=True,
            sort_within_batch=False,
            shuffle=shuffle,
        )
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.BucketIterator(
            repeat=False,
            dataset=dataset,
            batch_size=batch_size,
            batch_size_fn=batch_size_fn,
            train=False,
            sort=False,
            sort_within_batch=False
        )
    return data_iter


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(
            self,
            torch_batch,
            features_dim,
            is_train: bool = False,
            use_cuda: bool = False
    ):

        # Author Information
        self.id = torch_batch.id

        # Age
        self.age = torch_batch.age

        # Concentration
        self.concentration = torch_batch.concentration

        # Features
        self.features, self.features_length = torch_batch.features
        self.features_dim = features_dim
        self.features_mask = (self.features != torch.zeros(features_dim))[..., 0].unsqueeze(1)

        # Target
        self.target = torch_batch.target

        # Other
        self.use_cuda = use_cuda
        self.num_seqs = self.features.size(0)

        if use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """
        Move the batch to GPU
        :return:
        """
        self.features = self.features.cuda()
        self.features_mask = self.features_mask.cuda()

        self.concentration = self.concentration.cuda()
        self.age = self.age.cuda()

        if self.target is not None:
            self.target = self.target.cuda()