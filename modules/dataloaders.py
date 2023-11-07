import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
from .datasets import MIMICCXRDataset


class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle, drop_last):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split
        self.drop_last = drop_last

        if split == 'train':
            # self.transform = transforms.Compose([
            #     transforms.Resize((224, 224)),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.483, 0.483, 0.483),
            #                          (0.235, 0.235, 0.235))])
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.483, 0.483, 0.483),
                                     (0.235, 0.235, 0.235))])

        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.483, 0.483, 0.483),
                                     (0.235, 0.235, 0.235))])
            # self.transform = transforms.Compose([
            #     transforms.Resize((224, 224)),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.485, 0.456, 0.406),
            #                          (0.229, 0.224, 0.225))])

        if self.dataset_name == 'mimic_cxr':
            self.dataset = MIMICCXRDataset(self.args, self.tokenizer, self.split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'drop_last': self.drop_last
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        images_id, images, reports_ids, reports_masks, seq_lengths, toks, lengths = zip(*data)
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        max_length = max(lengths)
        targets_tok = np.zeros((len(toks), max_length), dtype=int)
        for i, tok in enumerate(toks):
            targets_tok[i, :len(tok)] = tok

        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks), torch.LongTensor(targets_tok)

