import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from utils import *
from data import test_dataloader
from skimage.metrics import peak_signal_noise_ratio
import time


def _eval(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=1)
    torch.cuda.empty_cache()
    adder = Adder()
    model.eval()
    with torch.no_grad():
        psnr_adder = Adder()
        # Hardware warm-up
        for iter_idx, (input_img, label_img, depth_img, _) in enumerate(dataloader):
            input_img = input_img.cuda()
            depth_img = depth_img.cuda()
            tm = time.time()
            _ = model(input_img, depth_img)
            _ = time.time() - tm

            if iter_idx == 20:
                break

        # Main Evaluation
        eval_start_time = time.time()
        for iter_idx, (input_img, label_img, depth_img, name) in enumerate(dataloader):
            input_img = input_img.cuda()
            depth_img = depth_img.cuda()

            tm = time.time()
            pred = model(input_img, depth_img)     # [2]
            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            if args.save_image:
                video_name, _, img_name = name[0].split('/')[-3:]
                save_name = os.path.join(args.result_dir, video_name, img_name)
                if not os.path.exists(os.path.join(args.result_dir, video_name)):
                    os.mkdir(os.path.join(args.result_dir, video_name))
                print("Saving %s" % save_name)
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)

            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr)
            print('%d iter PSNR: %.2f time: %f' % (iter_idx + 1, psnr, elapsed))

        print('==========================================================')
        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        print("Average time: %.4f" % adder.average())
        print("Total time: %.3f" % (time.time() - eval_start_time))


def _eval_self_ensemable(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    adder = Adder()
    model.eval()
    with torch.no_grad():
        psnr_adder = Adder()
        # Hardware warm-up
        for iter_idx,  (input_img, label_img, depth_img, name) in enumerate(dataloader):
            input_img = input_img.cuda()
            depth_img = depth_img.cuda()
            tm = time.time()
            _ = model(input_img, depth_img)
            _ = time.time() - tm
            if iter_idx == 20:
                break

        # Main Evaluation
        eval_start_time = time.time()
        for iter_idx, (input_img, label_img, depth_img, name) in enumerate(dataloader):
            input_img = input_img.cuda()
            depth_img = depth_img.cuda()
            input_imgs = self_ensemable(input_img, flip=False)
            depth_imgs = self_ensemable(depth_img, flip=False)
            pred_imgs = []
            elapsed = 0.
            for input_img, depth_img in enumerate(zip(input_imgs, depth_imgs)):
                tm = time.time()
                pred = model(input_img, depth_img)  # [2]
                elapsed += time.time() - tm
                pred_imgs.append(pred)

            pred = merge_multi_results(pred_imgs, flip=False)

            elapsed /= len(input_imgs)
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            if args.save_image:
                video_name, _, img_name = name[0].split('/')[-3:]
                save_name = os.path.join(args.result_dir, video_name, img_name)
                if not os.path.exists(os.path.join(args.result_dir, video_name)):
                    os.mkdir(os.path.join(args.result_dir, video_name))
                print("Saving %s" % save_name)
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)

            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr)
            print('%d iter PSNR: %.2f time: %f' % (iter_idx + 1, psnr, elapsed))

        print('==========================================================')
        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        print("Average time: %.4f" % adder.average())
        print("Total time: %.3f" % (time.time() - eval_start_time))