from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union
import pandas as pd

import h5py
import torch
from torch.utils.data import Dataset, DataLoader

__all__ = ["BagDataset"]


@dataclass
class BagDataset(Dataset):
    """A dataset of bags of instances"""

    bags: Sequence[Iterable[Path]]
    """The `.h5` files containing the bags

    Each bag consists of the features taken from one or multiple h5 files.
    Each of the h5 files needs to have a dataset called `feats` of shape N x F,
    where N is the number of instances and F the number of features per instance.
    """
    targets: Mapping[str, torch.Tensor]
    """The label of each bag"""
    instances_per_bag: Optional[int]
    """The number of instances to sample, or all samples if None"""
    deterministic: bool = True
    """Whether to sample deterministically
    
    If true, `instances_per_bag` samples will be taken equidistantly from the
    bag.  Otherwise, they will be sampled randomly.
    """

    extractor: Optional[str] = None
    """Feature extractor the features got extracted with

    Set on first encountered feature, if not set manually.  Will raise an error
    during runtime if features extracted with different feature extractors are
    encountered in the dataset.
    """

    def __len__(self):
        return len(self.bags)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns features, positions, targets"""

        # Collect features from all requested slides
        feat_list, coord_list = [], []
        for bag_file in self.bags[index]:
            with h5py.File(bag_file, "r") as f:
                # Ensure all features are created with the same feature extractor
                this_slides_extractor = f.attrs.get("extractor")
                if self.extractor is None:
                    self.extractor = this_slides_extractor
                assert this_slides_extractor == self.extractor, (
                    "all features have to be extracted with the same feature extractor! "
                    f"{bag_file} has been extracted with {this_slides_extractor}, "
                    f"expected {self.extractor}"
                )

                feats, coords = (
                    torch.tensor(f["feats"][:]),
                    torch.tensor(f["coords"][:]),
                )

            if self.instances_per_bag:
                feats, coords = pad_or_sample(
                    feats,
                    coords,
                    n=self.instances_per_bag,
                    deterministic=self.deterministic,
                )

            feat_list.append(feats.float())
            coord_list.append(coords.float())

        feats, coords = torch.concat(feat_list), torch.concat(coord_list)

        # We sample both on the slide as well as on the bag level
        # to ensure that each of the bags gets represented
        # Otherwise, drastically larger bags could "drown out"
        # the few instances of the smaller bags
        if self.instances_per_bag is not None:
            feats, coords = pad_or_sample(
                feats,
                coords,
                n=self.instances_per_bag,
                deterministic=self.deterministic,
            )

        return (
            feats,
            coords,
            {label: target.iloc[index] for label, target in self.targets.items()},
        )


def pad_or_sample(*xs: torch.Tensor, n: int, deterministic: bool) -> List[torch.Tensor]:
    assert (
        len(set(x.shape[0] for x in xs)) == 1
    ), "all inputs have to be of equal length"
    length = xs[0].shape[0]

    if length <= n:
        # Too few features; pad with zeros
        pad_size = n - length
        padded = [torch.cat([x, torch.zeros(pad_size, *x.shape[1:])]) for x in xs]
        return padded
    elif deterministic:
        # Sample equidistantly
        idx = torch.linspace(0, len(xs) - 1, steps=n, dtype=torch.int)
        return [x[idx] for x in xs]
    else:
        # Sample randomly
        idx = torch.randperm(length)[:n]
        return [x[idx] for x in xs]
    

def make_dataloaders(
    *,
    train_bags: Sequence[Iterable[Path]],
    train_targets: Mapping[str, torch.Tensor],
    valid_bags: Sequence[Iterable[Path]],
    valid_targets: Mapping[str, torch.Tensor],
    batch_size: int,
    instances_per_bag: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = BagDataset(
        bags=train_bags,
        targets=train_targets,
        instances_per_bag=instances_per_bag,
        deterministic=False,
    )
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )

    valid_ds = BagDataset(
        bags=valid_bags,
        targets=valid_targets,
        instances_per_bag=instances_per_bag,
        deterministic=True,
    )
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, num_workers=num_workers)

    return train_dl, valid_dl


def read_table(table: Union[Path, pd.DataFrame], dtype=str) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table

    if table.suffix == ".csv":
        return pd.read_csv(table, dtype=dtype, low_memory=False)  # type: ignore
    else:
        return pd.read_excel(table, dtype=dtype)  # type: ignore


def make_dataset_df(
    *,
    clini_table: Path,
    slide_table: Path,
    feature_dir: Path,
    patient_col: str = "PATIENT",
    filename_col: str = "FILENAME",
    group_by: Optional[str] = None,
    target_labels: Sequence[str],
) -> pd.DataFrame:

    slide_df = read_table(slide_table)
    slide_df = slide_df.loc[
        :, slide_df.columns.isin([patient_col, filename_col])  # type: ignore
    ]

    assert filename_col in slide_df, (
        f"{filename_col} not in {slide_table}. "
        "Use `--filename-col <COL>` to specify a different column name"
    )

    slide_df["path"] = slide_df[filename_col].apply(lambda fn: Path(f"{feature_dir}/{fn}.h5") \
                                                    if (Path(f"{feature_dir}/{fn}.h5")).exists() \
                                                        else None)
    if (na_idxs := slide_df.path.isna()).any():
        print(
            f"some slides from {slide_table} have no features: {list(slide_df.loc[na_idxs, filename_col])}",
            "warn",
        )
    slide_df = slide_df[~na_idxs]
    assert not slide_df.empty, f"no features for slide table {slide_table}"

    # df is now a DataFrame containing at least a column "path", possibly a patient and filename column


    assert patient_col in slide_df.columns, (
        f"a slide table with {patient_col} column has to be specified using `--slide-table <PATH>` "
        "or the patient column has to be specified with `--patient-col <COL>`"
    )

    clini_df = read_table(clini_table)
    # select all the relevant available ground truths,
    # make sure there's no conflicting patient info
    clini_df = (
        # select all important columns
        clini_df.loc[
            :, clini_df.columns.isin([patient_col, group_by, *target_labels])  # type: ignore
        ]
        .drop_duplicates()
        .set_index(patient_col, verify_integrity=True)
    )
    # TODO assert patient_col in clini_df, f"no column named {patient_col} in {clini_df}"
    df = slide_df.merge(clini_df.reset_index(), on=patient_col)
    assert not df.empty, "no match between slides and clini table"

    # At this point we have a dataframe containing
    # - h5 paths
    # - the corresponding slide names
    # - the patient id (if a slide table was given)
    # - the ground truths for the target labels present in the clini table

    group_by = group_by or patient_col if patient_col in df else filename_col

    # Group paths and metadata by the specified column
    grouped_paths_df = df.groupby(group_by)[["path"]].aggregate(list)
    grouped_metadata_df = (
        df.groupby(group_by)
        .first()
        .drop(columns=["path", filename_col], errors="ignore")
    )
    df = grouped_metadata_df.join(grouped_paths_df)

    return df