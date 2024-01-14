import torch
from dataset import EEGdenoiseNet
from enums import NoiseTypeEnum


def create_dataloader(
    batch_size,
    noise_types: list[NoiseTypeEnum],
):
    """Create dataloaders for EEG denoising.

    Two dataloaders are created, one for training and one for testing. The split is approx. 9:1.

    Parameters
    ----------
    batch_size : int
        Number of samples in each batch.
    noise_types : list[NoiseTypeEnum]
        List of noise types to mix with clean data.
    """
    ds_train = EEGdenoiseNet(
        eeg_data="data/EEG_all_epochs.npy",
        eog_data="data/EOG_all_epochs.npy",
        emg_data="data/EMG_all_epochs.npy",
        noise_types=noise_types,
        idx_int=(0, 4100),
    )
    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    ds_test = EEGdenoiseNet(
        eeg_data="data/EEG_all_epochs.npy",
        eog_data="data/EOG_all_epochs.npy",
        emg_data="data/EMG_all_epochs.npy",
        noise_types=noise_types,
        idx_int=(4100, None),
    )
    dl_test = torch.utils.data.DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )

    return dl_train, dl_test
