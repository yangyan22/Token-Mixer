import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length + 1  # Length_bos == 1 + max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]

    def __len__(self):
        return len(self.examples)


# class MimiccxrSingleImageDataset(BaseDataset):
#     def __getitem__(self, idx):
#         example = self.examples[idx]
#         image_id = example['id']
#         image_path = example['image_path']
#         image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
#         if self.transform is not None:
#             image = self.transform(image)
#         report_ids= self.tokenizer(example['report'])[:self.max_seq_length]
#         report_masks = [1] * len(report_ids)
#         seq_length = len(report_ids)
#         token = ""
#         for j in example['Tags']:
#             token = token + j + ". "
#         example['ids_tok'] = self.tokenizer(token+example['report'])[:self.max_seq_length+10]
#         length = len(example['ids_tok'])
#         tok_ids = example['ids_tok']
#         sample = (image_id, image, report_ids, report_masks, seq_length, tok_ids, length)
#         return sample
    
class MIMICCXRDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_paths']

        if self.transform is not None:
            if len(image_path)==1:
                image0 = self.transform(Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB'))
                image = image0
            elif len(image_path)==2:
                image0 = self.transform(Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB'))
                image1 = self.transform(Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB'))
                image = (image0 + image1)/2
            else:
                image0 = self.transform(Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB'))
                image1 = self.transform(Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB'))
                image2 = self.transform(Image.open(os.path.join(self.image_dir, image_path[2])).convert('RGB'))
                image = (image0 + image1 + image2)/3

        report_ids= self.tokenizer(example['token'])[:self.max_seq_length]
        report_masks = [1] * len(report_ids)
        seq_length = len(report_ids)
        token = ""
        for j in example['Tags']:
            token = token + j
        token = token +  ". "
        example['ids_tok'] = self.tokenizer(token+example['token'])[:self.max_seq_length+6]
        length = len(example['ids_tok'])
        tok_ids = example['ids_tok']
        sample = (image_id, image, report_ids, report_masks, seq_length, tok_ids, length)
        return sample
    

# class IuxrayMultiImageDataset(BaseDataset):
#     def __getitem__(self, idx):
#         example = self.examples[idx]
#         image_id = example['id']
#         image_path = example['image_path']
#         image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
#         image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
      
#         if self.transform is not None:
#             image_1 = self.transform(image_1)
#             image_2 = self.transform(image_2)

#         example['ids'] = self.tokenizer(example['report'])[:self.max_seq_length]
#         example['mask'] = [1] * len(example['ids'])
#         report_ids = example['ids']
#         report_masks = example['mask']
#         image = torch.stack((image_1, image_2), 0)
#         seq_length = len(report_ids)

#         token = ""
#         for j in example['MeSH']:
#             token = token + j + ". "
#         example['ids_tok'] = self.tokenizer(token+example['report'])[:self.max_seq_length+6]
#         length = len(example['ids_tok'])
#         tok_ids = example['ids_tok']
#         sample = (image_id, image, report_ids, report_masks, seq_length, tok_ids, length)
#         return sample



