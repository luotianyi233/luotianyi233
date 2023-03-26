import os
import torch
import time
from data import train_dataloader
from utils import Adder, Timer, check_lr, complex_depart
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
from torch.cuda.amp import autocast as autocast,GradScaler
import torch.nn.functional as F


def _train(model, args):
    #定义L1损失
    criterion = torch.nn.L1Loss()
    ##定义模型
    model=model.cuda()
    #定义Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,)
    ##在训练最开始之前实例化一个GradScaler对象
    scaler = GradScaler()
    #载入训练数据集
    dataloader = train_dataloader(args.data_dir, args.train_batch_size, args.train_batch_size)
    #获取迭代次数
    max_iter = len(dataloader)
    # 根据不同的学习率类型定义学习率的调度器
    if args.lr_type == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.lr_param[0], args.lr_param[1])
    elif args.lr_type == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_param[0], args.lr_param[1])
    else:
        exit(-1)
    # 如果设置了恢复，载入之前保存的训练状态
    epoch = 1
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        model.load_state_dict(state['model'])
        print('Resume from %d' % epoch)
        epoch += 1

    # 定义TensorBoard
    writer = SummaryWriter(args.writer_path)
    # 初始化Adder和Timer用于计算平均损失和耗时
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr=-1

    # 开始训练
    for epoch_idx in range(epoch, args.num_epoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        train_start_time = time.time()
        #迭代训练数据集种的图像
        for iter_idx, (input_img, label_img, depth_img) in enumerate(dataloader):
            # 将数据放到GPU上
            input_img = input_img.cuda()
            label_img = label_img.cuda()
            depth_img = depth_img.cuda()

            #梯度清零
            optimizer.zero_grad()
            ## 前向传播，通过模型生成模糊图像的预测，并计算预测与目标（清晰图像）之间的损失,同时开启autocast
            # 在模型执行推理时，由于models定义时的修改，在各cuda设备上的子进程中autocast生效
            # 在执行loss计算时，autocast生效
            with autocast():
                pred_img = model(input_img, depth_img)
                loss_content = criterion(pred_img, label_img)
            # label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
            # label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
            # l1 = criterion(pred_img[0], label_img4)
            # l2 = criterion(pred_img[1], label_img2)
            # l3 = criterion(pred_img[2], label_img)
            # loss_content = l1+l2+l3

            # label_fft1 = torch.rfft(label_img4, signal_ndim=2, normalized=False, onesided=False)
            # pred_fft1 = torch.rfft(pred_img[0], signal_ndim=2, normalized=False, onesided=False)
            # label_fft2 = torch.rfft(label_img2, signal_ndim=2, normalized=False, onesided=False)
            # pred_fft2 = torch.rfft(pred_img[1], signal_ndim=2, normalized=False, onesided=False)
            # label_fft3 = torch.rfft(label_img, signal_ndim=2, normalized=False, onesided=False)
            # pred_fft3 = torch.rfft(pred_img[2], signal_ndim=2, normalized=False, onesided=False)
            # label_fft1 = complex_depart(torch.fft.fft2(label_img4))
            # pred_fft1 = complex_depart(torch.fft.fft2(pred_img[0]))
            # label_fft2 = complex_depart(torch.fft.fft2(label_img2))
            # pred_fft2 = complex_depart(torch.fft.fft2(pred_img[1]))
            # label_fft3 = complex_depart(torch.fft.fft2(label_img))
            # pred_fft3 = complex_depart(torch.fft.fft2(pred_img[2]))

            # f1 = criterion(pred_fft1, label_fft1)
            # f2 = criterion(pred_fft2, label_fft2)
            # f3 = criterion(pred_fft3, label_fft3)
            # loss_fft = f1+f2+f3

            # 计算总损失并反向传播
            loss = loss_content     # + 0.01 * loss_fft
            #loss.backward()
            ##scales loss 为了梯度放大
            scaler.scale(loss).backward()
            #optimizer.step()
            ##scaler.step() 首先把梯度的值unscale回来
            ##如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,否则，忽略step调用，从而保证权重不更新（不被破坏）
            scaler.step(optimizer)

            ##准备着，看是否要增大scaler
            scaler.update()

            # 统计当前迭代的像素损失
            iter_pixel_adder(loss_content.item())
            # iter_fft_adder(loss_fft.item())

            # 统计整个 epoch 的像素损失
            epoch_pixel_adder(loss_content.item())
            # epoch_fft_adder(loss_fft.item())

            # 每 print_freq 个迭代输出一次状态，并将当前像素损失写入 Tensorboard
            if (iter_idx + 1) % args.print_freq == 0:
                lr = check_lr(optimizer)
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, lr, iter_pixel_adder.average(), 0))
                    # iter_fft_adder.average()))
                writer.add_scalar('DeblurNet/Loss_Pixel', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                # writer.add_scalar('DeblurNet/Loss_FFT', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                iter_timer.tic()
                iter_pixel_adder.reset()
                # iter_fft_adder.reset()
        # overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        # torch.save({'model': model.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'scheduler': scheduler.state_dict(),
        #             'epoch': epoch_idx}, overwrite_name)

        print(" Train time: %.3f" %(time.time() - train_start_time))

        # 每 save_freq 个 epoch 保存一次模型
        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, save_name)

        # 输出整个 epoch 的状态，包括所需时间和损失
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), 0.))  # epoch_fft_adder.average()))
        # epoch_fft_adder.reset()
        epoch_pixel_adder.reset()

        # 更新学习率
        scheduler.step()
        #检查当前epoch是否为指定的验证频率（valid_freq）的倍数，如果是，进行下一步操作，否则跳过。
        if epoch_idx % args.valid_freq == 0:
            #计算验证集上的平均PSNR和所有验证图像的重建图像，调用_valid函数实现此操作
            val_start_time = time.time()
            val_psnr, val_img = _valid(model, args, epoch_idx)
            #打印当前epoch，验证集上的平均PSNR值和验证时间，并输出分隔符"="
            print('%03d epoch \n Average PSNR %.3f dB' % (epoch_idx, val_psnr))
            print(" Val time: %.3f" % (time.time() - val_start_time))
            print('='*20)
            #使用TensorBoard记录验证集上的PSNR值和重建图像。
            writer.add_scalar('DeblurNet/PSNR_VAL', val_psnr, epoch_idx)
            writer.add_images('DeblurNet/IMG_VAL', val_img, epoch_idx)
            #如果当前的PSNR值比之前最佳的PSNR值更好，则更新最佳PSNR和对应的模型权重，并将其保存到指定目录下的文件中
            if val_psnr >= best_psnr:
                best_psnr = val_psnr
                torch.save({'model': model.state_dict(), 'best_psnr': best_psnr, 'epoch': epoch_idx},
                           os.path.join(args.model_save_dir, 'Best_%d_%.3f.pkl' % (epoch_idx, best_psnr)))
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    #定义最终训练得到的模型权重的保存路径，并将其保存到该路径下
    torch.save({'model': model.state_dict()}, save_name)
    #打印在验证集上得到的最佳PSNR值
    print('Best PSNR: ', best_psnr)
