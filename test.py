import argparse
from tqdm import tqdm
import torch
import csv
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.models_alter import Model
from modules.metrics import compute_scores
import os


def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='/public/home/yangyan/YangYan/images_downsampled',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='/public/home/jw12138/YZQ/mimic_all_paths.json',
                        help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr')
    parser.add_argument('--max_seq_length', type=int, default=100, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=10, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=4, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=64, help='the number of samples in a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet50', help='the visual extractor to be used.')  # resnet18 resnet50 resnet101 densenet121 pvt
    parser.add_argument('--pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')
    parser.add_argument('--tags', type=int, default=0, help='whether to concatenate the MeSH in report')
    parser.add_argument('--test_mode', type=str, default="sample_v", help='whether to concatenate the MeSH')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers_encoder', type=int, default=0, help='the number of layers of Transformer.')
    parser.add_argument('--num_layers_decoder', type=int, default=12, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the encoder output layer.')

    # Sample related  
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    parser.add_argument('--restore_dir', type=str, default='./results/res50clip_TD12_BS64_re5_seed9_losstlossv_lossret_lossrev_grandual_notags/', help='the path to load the models.')
    parser.add_argument('--checkpoint_dir', type=str, default='model_best.pth', help='the checkpont.')  
    parser.add_argument('--gn', type=str, default='_gn_bs3.csv')
    parser.add_argument('--n_gpu', type=int, default=2, help='the number of gpus to be used.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--gpus', type=str, default='0, 1', help='GPU IDs')
    parser.add_argument('--gpus_id', type=list, default=[0, 1], help='GPU IDs')
    args = parser.parse_args()
    return args


def main():
    args = parse_agrs()
    tokenizer = Tokenizer(args)
    model_path = os.path.join(args.restore_dir, args.checkpoint_dir)
    if args.n_gpu == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 0, 1, 2, 3 single-GPU
        device = torch.device('cuda:0')  # always: 0
        checkpoint = torch.load(model_path)
        print("Checkpoint loaded from epoch {}".format(checkpoint['epoch']))
        model = Model(args, tokenizer)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)  # the position of environ is important!
        print("GPUs_Used: {}".format(args.gpu))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus  # select multiple GPUs
        device = torch.device('cuda:0') # always: 0
        checkpoint = torch.load(model_path)
        print("Checkpoint loaded from epoch {}".format(checkpoint['epoch']))
        model = Model(args, tokenizer)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model = torch.nn.DataParallel(model, device_ids=args.gpus_id)  # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2" device_ids=[0, 1] 1 equals to GPU: 2
        print("GPUs_Used: {}".format(args.gpus))

    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False, drop_last=True)
    model.eval()
    with torch.no_grad():
        test_gts, test_res = [], []
        record_path_gt = os.path.join(args.restore_dir, args.dataset_name + '_gt.csv')
        record_path_gn = os.path.join(args.restore_dir, args.dataset_name + args.gn)
        if args.test_mode =="sample_v":
            with open(record_path_gt, "w", newline="") as f_gt:
                file_gt = csv.writer(f_gt)
                with open(record_path_gn, "w", newline="") as f_gn:
                    file_gn = csv.writer(f_gn)
                    with tqdm(desc='Epoch %d - Testing', unit='it', total=len(test_dataloader)) as pbar:
                        for batch_idx, (images_id, images, reports_ids, reports_masks, tok_ids) in enumerate(test_dataloader):
                            images, reports_ids, reports_masks, tok_ids = images.to(device), reports_ids.to(device), reports_masks.to(device), tok_ids.to(device)
                            output = model(images, targets=reports_ids, tok=tok_ids, mode='sample_v', tags=args.tags, epoch_id=0)
                            if args.n_gpu > 1:
                                reports = model.module.tokenizer.decode_batch(output.cpu().numpy())
                                ground_truths = model.module.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                                if args.tags ==1:
                                    tok = model.module.tokenizer.decode_batch(tok_ids[:, 1:].cpu().numpy())
                            else:
                                reports = model.tokenizer.decode_batch(output.cpu().numpy())
                                ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                                if args.tags ==1:
                                    tok = model.tokenizer.decode_batch(tok_ids[:, 1:].cpu().numpy())
                            test_gts.extend(ground_truths)  # gt for metrics
                            test_res.extend(reports)  # gn for metrics
                            pbar.update()
                            i = 0
                            for id in images_id:
                                print(id)
                                print('Predicted Sent.{}'.format(reports[i]))
                                print('Reference Sent.{}'.format(ground_truths[i]))
                                if args.tags ==1:
                                    print('Reference Sent.{}'.format(tok[i]))
                                print('\n')
                                file_gt.writerow([id, ground_truths[i]])
                                file_gn.writerow([id, reports[i]])
                                i = i + 1
            test_met = compute_scores({i: [gt] for i, gt in enumerate(test_gts)},
                                    {i: [re] for i, re in enumerate(test_res)})
            
        elif args.test_mode =="sample_t":
            with open(record_path_gt, "w", newline="") as f_gt:
                file_gt = csv.writer(f_gt)
                with open(record_path_gn, "w", newline="") as f_gn:
                    file_gn = csv.writer(f_gn)
                    with tqdm(desc='Epoch %d - Testing', unit='it', total=len(test_dataloader)) as pbar:
                        for batch_idx, (images_id, images, reports_ids, reports_masks, tok_ids) in enumerate(test_dataloader):
                            images, reports_ids, reports_masks, tok_ids = images.to(device), reports_ids.to(device), reports_masks.to(device), tok_ids.to(device)
                            output = model(images, targets=reports_ids, tok=tok_ids, mode='sample_t', tags=args.tags, epoch_id=0)
                            if args.n_gpu > 1:
                                reports = model.module.tokenizer.decode_batch(output.cpu().numpy())
                                ground_truths = model.module.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                                if args.tags ==1:  # if tags ==1, the input for report reconstruction is concatenated with the tags. And the result is different from tags==0.
                                    tok = model.module.tokenizer.decode_batch(tok_ids[:, 1:].cpu().numpy())
                            else:
                                reports = model.tokenizer.decode_batch(output.cpu().numpy())
                                ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                                if args.tags ==1:
                                    tok = model.tokenizer.decode_batch(tok_ids[:, 1:].cpu().numpy())
                            test_gts.extend(ground_truths)  # gt for metrics
                            test_res.extend(reports)  # gn for metrics
                            pbar.update()
                            i = 0
                            for id in images_id:
                                print(id)
                                print('Predicted Sent.{}'.format(reports[i]))
                                print('Reference Sent.{}'.format(ground_truths[i]))
                                if args.tags ==1:
                                    print('Reference Sent.{}'.format(tok[i]))
                                print('\n')
                                file_gt.writerow([id, ground_truths[i]])
                                file_gn.writerow([id, reports[i]])
                                i = i + 1
            test_met = compute_scores({i: [gt] for i, gt in enumerate(test_gts)},
                                    {i: [re] for i, re in enumerate(test_res)})
        print(test_met)



if __name__ == '__main__':
    main()
