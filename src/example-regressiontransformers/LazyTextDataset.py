import linecache
from pathlib import Path

from pylt3.utils.file_helpers import get_n_lines
from torch.utils.data import Dataset


class LazyTextDataset(Dataset):
    """ Dataset class to use in Dataset/DataLoader setup. Expected input:
        {'sentences': '/path/to/sentences.txt',
         'labels': '/path/to/labels.txt'}
        One sentence/label per line.
    """

    def __init__(self, paths_obj):
        self.sentences_f = (
            str(Path(paths_obj["sentences"]).resolve())
            if "sentences" in paths_obj
            else None
        )
        self.labels_f = str(Path(paths_obj["labels"]).resolve())

        self.num_entries = get_n_lines(self.sentences_f)
        if self.num_entries != get_n_lines(self.labels_f):
            raise ValueError(
                "Number of lines not identical between sentences and labels"
            )

    def __getitem__(self, idx):
        # linecache starts counting from one, not zero
        idx += 1
        d = {
            "sentences": linecache.getline(self.sentences_f, idx),
            "labels": linecache.getline(self.labels_f, idx),
        }

        return d

    def __len__(self):
        return self.num_entries
