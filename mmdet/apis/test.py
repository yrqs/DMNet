import os.path as osp
import pickle
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

import tqdm
import time

# idx_list = [20, 38, 42, 43]
idx_list = None

def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            if idx_list is None or i in idx_list:
                model.module.show_result(data, result, score_thr=0.2, idx=i)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # test = model.backbone(**data)
            # print(dir(model))
            result = model(return_loss=False, rescale=True, **data)
            # print([r.shape for r in result])
            # print(result)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def multi_gpu_pre_test(model, data_loader):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    dataset = data_loader.dataset
    dataset.pre_test = True
    rank, world_size = get_dist_info()
    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    model.module.bbox_head.meta_representatives *= 0
    model.module.bbox_head.meta_representatives_counter *= 0

    dist.broadcast(model.module.bbox_head.meta_representatives, src=0)
    dist.broadcast(model.module.bbox_head.meta_representatives_counter, src=0)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            model(return_loss=False, rescale=True, pre_test=True, **data)

    meta_representatives_counter_gather = [
        torch.zeros_like(model.module.bbox_head.meta_representatives_counter).to(model.module.bbox_head.meta_representatives_counter.device) for
        _ in range(world_size)]
    meta_representatives_gather = [
        torch.zeros_like(model.module.bbox_head.meta_representatives).to(model.module.bbox_head.meta_representatives.device) for
        _ in range(world_size)]
    dist.all_gather(meta_representatives_counter_gather, model.module.bbox_head.meta_representatives_counter)
    dist.all_gather(meta_representatives_gather, model.module.bbox_head.meta_representatives)
    meta_representatives_counter_gather = [m.to(meta_representatives_counter_gather[0].device) for m in meta_representatives_counter_gather]
    meta_representatives_gather = [m.to(meta_representatives_gather[0].device) for m in meta_representatives_gather]
    meta_representatives_counter_gather = torch.stack(meta_representatives_counter_gather).sum(0)
    meta_representatives_gather = torch.stack(meta_representatives_gather).sum(0)
    # print('meta_representatives_counter_gather: ', meta_representatives_counter_gather)
    # assert meta_representatives_gather[meta_representatives_counter_gather==0].sum() == 0
    meta_representatives_counter_gather[meta_representatives_counter_gather==0] = 1
    meta_representatives = meta_representatives_gather / meta_representatives_counter_gather.unsqueeze(-1)
    model.module.bbox_head.meta_representatives.data = meta_representatives
    dist.broadcast(model.module.bbox_head.meta_representatives, src=0)
    # print(model.module.bbox_head.meta_representatives)


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
