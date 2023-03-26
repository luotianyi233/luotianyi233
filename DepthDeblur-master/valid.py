import torch
from data import valid_dataloader
from utils import Adder, PSNR
import os


def _valid(model, args, ep):
    dataset = valid_dataloader(args.data_dir, args.test_batch_size, num_workers=args.test_batch_size)
    model.eval()
    psnr_adder = Adder()

    with torch.no_grad():
        print('Start Evaluation')
        for idx, (input_img, label_img, depth_img) in enumerate(dataset):
            input_img = input_img.cuda()
            depth_img = depth_img.cuda()
            label_img = label_img.cuda()
            if not os.path.exists(os.path.join(args.result_dir, '%d' % ep)):
                os.mkdir(os.path.join(args.result_dir, '%d' % ep))

            pred = model(input_img, depth_img)
            pred_clip = torch.clamp(pred, 0, 1)
            # pred_clip = torch.clamp(pred[2], 0, 1)

            psnr = PSNR(pred_clip, label_img)
            # p_numpy = pred_clip.squeeze(0).cpu().numpy()
            # label_numpy = label_img.squeeze(0).cpu().numpy()
            # psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)

            psnr_adder(psnr)
            if idx % 100 == 0:
                print('\r%03d' % idx, end=' ')

    # print('\n')
    model.train()
    return psnr_adder.average(), (pred_clip.detach().cpu()[0]).unsqueeze(0)