#!/usr/bin/python
# -*- coding: UTF-8 -*-
# author：Mengliang Zhu

from omegaconf import open_dict
from pathlib import Path
import torch, os, wandb, omegaconf, logging, hydra, copy
import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from Lib.train_utils import cal_acc, AverageMeter, seed_torch, cal_acc_P
from Lib.model import get_model, get_opt_parameters


# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

class TTA_Trainer():
    def __init__(self, cfg: omegaconf.DictConfig, run: wandb.sdk.wandb_run.Run):
        seed_torch(cfg.seed_run)
        with open_dict(cfg):
            if cfg.Model.model_type == 'linear':
                cfg.model_name = str(cfg.Model.model_name) + str(cfg.seed_run) + cfg.Dataset.input_kind + '_Linear.pt'
            elif cfg.Model.model_type == 'wn':
                cfg.model_name = str(cfg.Model.model_name) + str(cfg.seed_run) + cfg.Dataset.input_kind + '_WN.pt'
            else:
                cfg.model_name = str(cfg.Model.model_name) + str(cfg.seed_run) + cfg.Dataset.input_kind + '_Pro.pt'
            cfg.save_model_path = r'.\TTA_Model'
            cfg.model_path = Path(cfg.save_model_path) / (cfg.Dataset.data_name + str(cfg.Opt.lr_src)) / (
                        str(cfg.Dataset.TL_Task) + '_Task')
            Path(cfg.model_path).mkdir(parents=True, exist_ok=True)
            if cfg.PR > 0 and cfg.PR < 1:
                cfg.model_name = str(cfg.PR) + '_' + cfg.model_name
                cfg.model_txt = Path(cfg.model_path) / (cfg.model_name[0:-3] + '.txt')
                with open(cfg.model_txt, "r") as f:
                    cfg_model = f.readline()
                    cfg_model = cfg_model[1:-2].split(',')
                    cfg_model = [int(s) for s in cfg_model]
                    cfg.cfg_model = cfg_model

        self.cfg = cfg
        self.run = run

    def setup(self):
        cfg = self.cfg
        seed_torch(cfg.seed_run)
        import warnings
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert cfg.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        dataset = getattr(Dataset, cfg.Dataset.data_name)
        self.dataset = dataset
        self.num_classes = dataset.num_classes

        self.datasets = {}
        self.datasets['source_data'], self.datasets['target_data'] \
            = dataset(**cfg.Dataset).data_generator()
        
        # for test
        self.dataloaders = {x: DataLoader(self.datasets[x], batch_size=cfg.batch_size,
                                          shuffle=False, num_workers=cfg.num_workers, drop_last=False,
                                          pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['source_data', 'target_data']}
        # for training
        self.target_dataloader = DataLoader(self.datasets['target_data'], batch_size=cfg.batch_size,
                                            shuffle=True, num_workers=cfg.num_workers,
                                            pin_memory=(True if self.device == 'cuda' else False),
                                            drop_last=False)
        # for training
        self.source_dataloader = DataLoader(self.datasets['source_data'], batch_size=cfg.batch_size,
                                            shuffle=True, num_workers=cfg.num_workers,
                                            pin_memory=(True if self.device == 'cuda' else False),
                                            drop_last=False)
        if cfg.PR > 0 and cfg.PR < 1:
            # load pruned source model with lightweight DNN architecture
            self.model = get_model(num_classes=self.num_classes, cfg=cfg.cfg_model, **cfg.Model).to(self.device)
        else:
            # load the original source model with high-capacity DNN architecture
            self.model = get_model(num_classes=self.num_classes, **cfg.Model)
    
    # online target model adaptation
    def OTMA(self):
        cfg = self.cfg
        seed_torch(cfg.seed_run)
        import TTA_Utils.tta as tta
        model = self.model.to(self.device)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(Path(cfg.model_path) / cfg.model_name))
        else:
            model.load_state_dict(torch.load(Path(cfg.model_path) / cfg.model_name), map_location='cpu')
        acc_t_tar = cal_acc(self.dataloaders["target_data"], model)[0]
        if cfg.process_wandb:
            self.run.log({'acc_iter': acc_t_tar}, step=0)
            print(f'Task: {cfg.Dataset.TL_Task}: Beginning Acc T = {acc_t_tar:.2f}%; ')
            
        model = tta.configure_model_BN(model)
        params, param_names = tta.collect_params_BN(model)
        optimizer = get_opt_parameters(params, cfg.Opt, kind='tar')
        model_tar = tta.TTA(model, optimizer, device=self.device, **cfg.TTA)
        model_tar.reset()
        epoch_num = 0
        iter_num = 0
        interval_iter = len(self.target_dataloader)
        max_iter = interval_iter

        Acc_tar_meter = AverageMeter()
        Acc_uncertain_meter = AverageMeter()
        Acc_certain_meter = AverageMeter()

        while iter_num < max_iter:
            iter_num += 1
            try:
                X_T, Y_T, _ = iter_target.next()
            except:
                iter_target = iter(self.target_dataloader)
                X_T, Y_T, _ = iter_target.next()

            if X_T.size(0) == 1:
                continue

            X_T, Y_T = X_T.to(self.device), Y_T.to(self.device)
            n_t = X_T.size()[0]
            
            acc_t_tar, acc_certain, acc_uncertain, loss = model_tar(X_T, Y_T)
            
            Acc_certain_meter.update(acc_certain.cpu().detach().numpy())
            Acc_uncertain_meter.update(acc_uncertain.cpu().detach().numpy())
            Acc_tar_meter.update(acc_t_tar, n_t)


            if cfg.process_wandb:
                self.run.log({'acc_iter': acc_t_tar}, step=iter_num)
                self.run.log({'acc_certain': acc_certain}, step=iter_num)
                self.run.log({'acc_uncertain': acc_uncertain}, step=iter_num)
                self.run.log({'loss': loss.cpu().detach().numpy()}, step=iter_num)


            if iter_num == max_iter:
                epoch_num += 1
                acc_t = Acc_tar_meter.avg
                acc_uncertain_ = Acc_uncertain_meter.avg
                acc_certain_ = Acc_certain_meter.avg
                Acc_tar_meter.reset()
                Acc_certain_meter.reset()
                Acc_uncertain_meter.reset()
                log_str = 'Task: {}, Epoch:{}/{}; Acc_tar/uncertain/certain = {:.2f}%/{:.2f}%/{:.2f}%; ' \
                    .format(cfg.Dataset.TL_Task, epoch_num, cfg.train_epoch, acc_t, acc_uncertain_, acc_certain_)

                from thop import profile
                from TTA_Utils.utils import getModelSize
                x = torch.randn((128, 1, 512)).to(self.device)
                model_size = getModelSize(model)[-1]
                flops, params_num = profile(model, (x,), verbose=False)
                flops = flops/128
                print("the ResNet8: modelsize {:.3f}MB flops {} and params {}".format(model_size, flops, params_num))

                if cfg.process_wandb:
                    self.run.log({'acc_t': acc_t}, step=iter_num)
                    self.run.log({'acc_uncertain_avg': acc_uncertain_}, step=iter_num)
                    self.run.log({'acc_certain_avg': acc_certain_}, step=iter_num)
                    self.run.log({'model_size': model_size}, step=iter_num)
                    self.run.log({'flops': flops}, step=iter_num)
                    self.run.log({'params': params_num}, step=iter_num)
                if cfg.verbose:
                    print(log_str + '\n')


@hydra.main(version_base=None, config_path='./Configs', config_name='defaults')
def run(cfg: omegaconf.DictConfig):
    from itertools import permutations
    TL_Task_list = list(permutations(cfg.Dataset.TL_list, 2))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    import time
    train_time = time.strftime('%m-%d %H:%M', time.localtime(time.time()))
    seed_list = list(range(2025, 2026))

    for seed_run in seed_list:
        for TL_Task in TL_Task_list:
            with (open_dict(cfg)):
                # for wandb settings
                cfg.train_time = train_time
                cfg.seed_run = seed_run
                cfg.Dataset.TL_Task = TL_Task
                cfg.wandb.setup.name = str(TL_Task) + 'Task_' + str(cfg.seed_run)
                wandb_cfg = omegaconf.OmegaConf.to_container(
                    cfg, resolve=True, throw_on_missing=True
                )
                cfg.train_epoch = 1  # for online target model adaptation
                cfg.Dataset.input_kind = 'fft'
                cfg.Model.bottleneck_num = 128
                cfg.batch_size = 128
                if cfg.Dataset.data_name == 'Gear':
                    cfg.Opt.lr_src = 1e-3
                    cfg.Opt.lr_tar = 1e-2
                elif cfg.Dataset.data_name == 'CWRU':
                    cfg.Opt.lr_src = 1e-3
                    cfg.Opt.lr_tar = 1e-2
                elif cfg.Dataset.data_name == 'PU':
                    cfg.Opt.lr_src = 1e-3
                    cfg.Opt.lr_tar = 1e-2
                cfg.verbose = True
                cfg.Opt.weight_decay_tar = cfg.Opt.lr_tar / 10
                cfg.TTA.temp = 1
                cfg.TTA.optim_steps = 2
                cfg.model_episodic = False
                cfg.TTA.B_kind = 1  # B_kind=2的效果在CWRU上不好

                #关键的超参数
                cfg.TTA.contrastive_ = 1
                cfg.TTA.NA_ = 1
                cfg.TTA.alpha = 2
                cfg.TTA.neighbor_K = 5
                cfg.TTA.filter_K = 50
                cfg.TTA.NA_kind = 4
                cfg.PR = 0.6
                cfg_based = cfg.Dataset.input_kind + '_' + cfg.Model.model_name + '_' + str(cfg.PR) + cfg.Opt.name \
                            + '_lr_' + str(cfg.Opt.lr_tar) + '_BS_' + str(cfg.batch_size) + '_' + str(cfg.TTA.version)
                cfg_opt = 'filter_K' + str(cfg.TTA.filter_K)  + '_alpha_' + str(cfg.TTA.alpha)+\
                '_NK_' + str(cfg.TTA.neighbor_K)
                group_name = train_time + cfg_based + cfg_opt
                cfg.wandb.setup.group = group_name
                cfg.wandb.setup.project = "TTA_tar" + '_' + cfg.Dataset.data_name
                run = wandb.init(config=wandb_cfg, **cfg.wandb.setup)

                with run:
                    TTA = TTA_Trainer(cfg, run)
                    TTA.setup()
                    print(f"--------{cfg.Dataset.data_name} Dataset:{TL_Task} TL Task---------")
                    print(f"the TTA infor is {cfg.TTA}")
                    print(f"the input kind is {cfg.Dataset.input_kind},"
                          f"the model kind is {cfg.Model.model_name}")
                    cfg_dict = omegaconf.OmegaConf.to_container(cfg)
                    print(f"the opt infor is {cfg_dict['Opt']}")
                    print(f"the data infor is {cfg_dict['Dataset']}")
                    cfg.process_wandb = True
                    TTA.OTMA()

        from pprint import pprint
        print(f"the TL task is {TL_Task_list}")
        pprint(f"the config information is {cfg_dict}")


if __name__ == '__main__':
    run()




