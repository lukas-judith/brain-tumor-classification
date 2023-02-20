import torch as tc

from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, data, labels):
        super(ImageDataset, self).__init__()

        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = tc.tensor(self.data[index], dtype=tc.float32)
        y = tc.tensor(self.labels[index], dtype=tc.float32)
        return x, y


def get_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)



