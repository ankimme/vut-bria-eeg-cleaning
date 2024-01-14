#!/usr/bin/env python3

import numpy as np
from torch.utils.data import Dataset
import random
from enums import NoiseTypeEnum


class EEGdenoiseNet(Dataset):
    def __init__(
        self,
        eeg_data: str,
        eog_data: str,
        emg_data: str,
        noise_types: list[NoiseTypeEnum],
        idx_int: tuple[int, int] = (None, None),
    ):
        """Create EEG denoising dataset.

        Parameters
        ----------
        eeg_data : str
            Path to clean EEG data.
        eog_data : str
            Path to EOG artefacts data.
        emg_data : str
            Path to EMG artefacts data.
        noise_types : list[NoiseTypeEnum]
            List of noise types to mix with clean data.
        idx_int : tuple[int, int], optional
            Interval of EEG data to use, take all values by default
        """
        self._eeg_data = np.load(eeg_data)[idx_int[0] : idx_int[1]]
        self._eog_data = np.load(eog_data)
        self._emg_data = np.load(emg_data)
        self._noise_types = noise_types
        self._eeg_data = self._eeg_data.astype(np.float32)
        self._eog_data = self._eog_data.astype(np.float32)
        self._emg_data = self._emg_data.astype(np.float32)

    def __len__(self) -> int:
        return len(self._eeg_data)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, NoiseTypeEnum]:
        clean_eeg_segment = self._eeg_data[index]

        # get random noise type
        category = random.choice(self._noise_types)
        combined_eeg_segments = self._add_noise(clean_eeg_segment, category)
        combined = self._normalize(combined_eeg_segments)
        clean = self._normalize(clean_eeg_segment)
        return combined, clean, category.value

    def _add_noise(
        self, clean_segment: np.ndarray, noise_type: NoiseTypeEnum
    ) -> np.ndarray:
        """Add noise to clean EEG segments. SNR and noise sample are chosen randomly."""
        match noise_type:
            case NoiseTypeEnum.CLEAN:
                return clean_segment
            case NoiseTypeEnum.EYE_MOVEMENT:
                snr_db = random.uniform(-7, 2)  # according to paper
                eog_index = random.randrange(0, len(self._eog_data))
                noisy_segment = self._eog_data[eog_index]
                return self._combine_clean_and_noisy_segments(
                    clean_segment, noisy_segment, snr_db
                )
            case NoiseTypeEnum.FACIAL_MUSCLES_MOVEMENT:
                snr_db = random.uniform(-7, 4)  # according to paper
                emg_index = random.randrange(0, len(self._emg_data))
                noisy_segment = self._emg_data[emg_index]
                return self._combine_clean_and_noisy_segments(
                    clean_segment, noisy_segment, snr_db
                )
            case _:
                raise ValueError(f"Unknown noise type: {noise_type}")

    def _rms(self, records: np.ndarray) -> float:
        """Calculate root mean square of data."""
        return np.sqrt(np.mean(np.square(records)))

    def _normalize(self, records: np.ndarray) -> np.ndarray:
        """Normalize data."""
        return records / np.std(records)

    def _combine_clean_and_noisy_segments(
        self, clean_segments: np.ndarray, noisy_segments: np.ndarray, snr_db: float
    ) -> np.ndarray:
        """Combine clean and noisy EEG segments. Using methods from paper."""
        snr_train = 10 ** (0.1 * snr_db)

        coe = self._rms(clean_segments) / (self._rms(noisy_segments) * snr_train)
        adjusted_noise = noisy_segments * coe
        result = clean_segments + adjusted_noise
        return result
