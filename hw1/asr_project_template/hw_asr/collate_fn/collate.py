import logging
from typing import List, Dict, Union
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]) -> Dict[str, Union[List[str], torch.Tensor]]:
    """
    Collate and pad fields in dataset items
    """
    result_batch = {}
    result_batch["spectrogram"] = torch.zeros(
        len(dataset_items),
        dataset_items[0]["spectrogram"].shape[1],
        max([dataset_item["spectrogram"].shape[2] for dataset_item in dataset_items])
    )
    for ind, dataset_item in enumerate(dataset_items):
        result_batch["spectrogram"][ind, :, :dataset_item["spectrogram"].shape[2]] = dataset_item["spectrogram"]
    result_batch["text_encoded"] = torch.zeros(
        len(dataset_items),
        max([dataset_item["text_encoded"].shape[1] for dataset_item in dataset_items])
    )
    for ind, dataset_item in enumerate(dataset_items):
        result_batch["text_encoded"][ind, :dataset_item["text_encoded"].shape[1]] = dataset_item["text_encoded"]
    result_batch["text"] = [dataset_item["text"] for dataset_item in dataset_items]
    result_batch["audio_path"] = [dataset_item["audio_path"] for dataset_item in dataset_items]
    result_batch["text_encoded_length"] = torch.IntTensor([dataset_item["text_encoded"].shape[1] for dataset_item in dataset_items])
    result_batch["spectrogram_length"] = torch.IntTensor([dataset_item["spectrogram"].shape[2] for dataset_item in dataset_items])
    
    return result_batch
    
    
