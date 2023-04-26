import glob
import os

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_
from pcdet.models import load_data_to_gpu
import pdb
import numpy as np

def cal_mimic_loss(batch_teacher, batch, model_cfg, mimic_mode):
    teacher_features = batch_teacher['spatial_features_2d'].detach()
    student_features = batch['spatial_features_2d']

    if mimic_mode == 'gt':
        rois = batch['gt_boxes']
    else:
        rois = batch_teacher['rois_mimic'].detach()

    dataset_cfg = batch['dataset_cfg']
    min_x = dataset_cfg.POINT_CLOUD_RANGE[0]
    min_y = dataset_cfg.POINT_CLOUD_RANGE[1]
    voxel_size_x = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[0]
    voxel_size_y = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[1]
    down_sample_ratio = model_cfg.ROI_GRID_POOL.DOWNSAMPLE_RATIO


    if mimic_mode in ['roi', 'gt']:
        batch_size, height, width = teacher_features.size(0), teacher_features.size(2), teacher_features.size(3)
        roi_size = rois.size(1)

        x1 = (rois[:, :, 0] - rois[:, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
        x2 = (rois[:, :, 0] + rois[:, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
        y1 = (rois[:, :, 1] - rois[:, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)
        y2 = (rois[:, :, 1] + rois[:, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)
        #print(height, width,x1.min(),x2.max(),y1.min(),y2.max())
        mask = torch.zeros(batch_size, roi_size, height, width).bool().cuda()
        grid_y, grid_x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
        grid_y = grid_y[None, None].repeat(batch_size, roi_size, 1, 1).cuda()
        grid_x = grid_x[None, None].repeat(batch_size, roi_size, 1, 1).cuda()

        mask_y = (grid_y >= y1[:, :, None, None]) * (grid_y <= y2[:, :, None, None])
        mask_x = (grid_x >= x1[:, :, None, None]) * (grid_x <= x2[:, :, None, None])
        mask = (mask_y * mask_x).float()
        if mimic_mode == 'gt':
            mask[rois[:,:,-1] == 0] = 0
        weight = mask.sum(-1).sum(-1) #bz * roi
        weight[weight == 0] = 1
        mask = mask / weight[:, :, None, None]

        mimic_loss = torch.norm(teacher_features - student_features, p=2, dim=1)

        mask = mask.sum(1)
        mimic_loss = (mimic_loss * mask).sum() / batch_size / roi_size
        if mimic_mode == 'gt':
            mimic_loss = (mimic_loss * mask).sum() / (rois[:,:,-1] > 0).sum()
    elif mimic_mode == 'all':
        mimic_loss = torch.mean(torch.norm(teacher_features - student_features, p=2, dim=1))
    else:
        raise NotImplementedError

    return mimic_loss



def train_one_epoch(model, model_teacher, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, mimic_weight=1, mimic_mode='roi'):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)


    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            batch, batch_teacher = next(dataloader_iter)

        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch, batch_teacher = next(dataloader_iter)

            print('new iters')

        #batch_teacher = batch.copy()
        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        model_teacher.eval()
        optimizer.zero_grad()

        load_data_to_gpu(batch_teacher)
        batch_teacher['mimic'] = 'mimic'
        batch['mimic'] = 'mimic'
        with torch.no_grad():
            tb_dict_teacher, disp_dict_teacher, batch_teacher_new = model_teacher(batch_teacher)


        # batch['rois_mimic'] = batch_teacher_new['rois_mimic'].clone()
        temp, batch_new = model_func(model, batch)
        loss, tb_dict, disp_dict = temp
        loss_mimic = cal_mimic_loss(batch_teacher_new, batch_new, model.module.model_cfg.ROI_HEAD, mimic_mode)
        loss_sum = loss + loss_mimic * mimic_weight

        loss_sum.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'loss_mimic': loss_mimic.item(), 'lr': cur_lr})
        #disp_dict.update({'loss': loss.item(),  'lr': cur_lr})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model_mimic(model, model_teacher, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, ps_label_dir,
                source_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, logger=None, ema_model=None, mimic_weight=1, mimic_mode='roi'):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)

        for cur_epoch in tbar:
            if source_sampler is not None:
                source_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, model_teacher, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                mimic_weight=mimic_weight,
                mimic_mode=mimic_mode
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
