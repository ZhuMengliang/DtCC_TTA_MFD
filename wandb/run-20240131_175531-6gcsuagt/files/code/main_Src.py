#!/usr/bin/python
# -*- coding: UTF-8 -*-
# author：Mengliang Zhu
from omegaconf import open_dict, OmegaConf
from pathlib import Path
import torch, os, wandb, omegaconf, logging, hydra, copy
import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from Lib.train_utils import AverageMeter, cal_acc, seed_torch, cal_acc_P
from Lib.losses import CrossEntropyLabelSmooth

from Lib.model import get_model
from Lib.optimizer import get_optimizer, get_lr_scheduler
import numpy as np

class Source_Trainer():
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
            cfg.model_path = Path(cfg.save_model_path) / (cfg.Dataset.data_name + str(cfg.Opt.lr_src)) / (str(cfg.Dataset.TL_Task) + '_Task')
            Path(cfg.model_path).mkdir(parents=True, exist_ok=True)
 
            cfg.verbose = True

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
                                          shuffle=False, num_workers=cfg.num_workers,
                                          pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['source_data', 'target_data']}
        
        # for training
        self.target_dataloader = DataLoader(self.datasets['target_data'], batch_size=cfg.batch_size,
                                            shuffle=True, num_workers=cfg.num_workers,
                                            pin_memory=(True if self.device == 'cuda' else False),
                                            drop_last=False)
        #for training
        self.source_dataloader = DataLoader(self.datasets['source_data'], batch_size=cfg.batch_size,
                                            shuffle=True, num_workers=cfg.num_workers,
                                            pin_memory=(True if self.device == 'cuda' else False),
                                            drop_last=False)

        self.src_model = get_model(num_classes=self.num_classes, **cfg.Model)
    
    # source model training
    def SMT(self, model=None):
        cfg = self.cfg
        seed_torch(cfg.seed_run)
        if model is None:
            model = self.src_model.to(self.device)
            model_PR = False
            # source model training for original high-capacity DNN model
        else:
            model_PR = True
            # source model training for pruned DNN model with lightweight architecture
        optimizer = get_optimizer(model, args=cfg.Opt, kind='src')
        lr_scheduler = get_lr_scheduler(optimizer, args=cfg.Opt, epoch_train=cfg.src_epoch, kind='src')
        interval_iter = len(self.source_dataloader)
        max_iter = cfg.src_epoch * interval_iter

        iter_num = 0
        index = 0
        loss_meter = AverageMeter()
        while iter_num < max_iter:
            try:
                X_S, Y_S, _ = iter_source.next()
            except:
                iter_source = iter(self.source_dataloader)
                X_S, Y_S, _ = iter_source.next()
            if X_S.size(0) == 1:
                continue

            iter_num += 1
            model.train()
            X_S, Y_S = X_S.to(self.device), Y_S.to(self.device)
            batch = X_S.size()[0]
            Y_pre = model(X_S)
            loss_cls = CrossEntropyLabelSmooth(num_classes=self.num_classes, epsilon=0.1) \
                (Y_pre, Y_S)
            optimizer.zero_grad()
            loss_cls.backward()
            optimizer.step()
            loss_meter.update(loss_cls.cpu().detach().numpy(), batch)

            if iter_num % interval_iter == 0 or iter_num == max_iter:
                index += 1
                model.eval()
                acc_s = cal_acc(self.dataloaders["source_data"], model, self.device)[0]
                acc_t = cal_acc(self.dataloaders["target_data"], model, self.device)[0]
                loss_s = loss_meter.avg
                loss_meter.reset()

                log_str = 'Task: {}, Iter:{}/{}; Accuracy_S/T = {:.2f}%/ {:.2f}%; Loss= {:.2f}' \
                    .format(cfg.Dataset.TL_Task, iter_num // len(self.source_dataloader), cfg.src_epoch, acc_s,
                            acc_t, loss_s)
                if cfg.process_wandb:
                    self.run.log({'loss_s': loss_s}, step=index)
                    self.run.log({'acc_s': acc_s, 'acc_t': acc_t}, step=index)

                if cfg.verbose:
                    print(log_str + '\n')
                elif iter_num == max_iter:
                    print(log_str + '\n')
                model.train()
                if lr_scheduler is not None:
                    lr_scheduler.step()

        self.model_src = model
        
        if cfg.save_models and not model_PR:
            torch.save(model.state_dict(), Path(cfg.model_path) / cfg.model_name)
        elif cfg.save_models and model_PR:
            model_name = str(cfg.PR) + '_' + cfg.model_name
            torch.save(model.state_dict(), Path(cfg.model_path) / model_name)
            model_txt = Path(cfg.model_path) / (model_name[0:-3] + '.txt')
            with open(model_txt, 'w+') as file:
                file.write(str(model[0].cfg_model))
                file.write('\n')
                file.write('------------------------------------\n')
                # record the detailed information about the architecture of pruned model
                
        return acc_s, acc_t

    def SMP(self):  # source model pruning
        from Lib.model import get_model
        cfg = self.cfg
        seed_torch(cfg.seed_run)
        if cfg.load_models:
            model_src = get_model(num_classes=self.num_classes, **cfg.Model).to(self.device)
            model_src.load_state_dict(torch.load(Path(cfg.model_path) / cfg.model_name))
        else:
            model_src = self.model_src
        acc_t = cal_acc(self.dataloaders["target_data"], model_src, self.device)[0]
        print(f"the initial accc is {acc_t:.2f}")

        from Model_Zoos.Prune import Model_Prune
        cfg_model, cfg_mask = Model_Prune(model_src, cfg, self)
        model_new = get_model(num_classes=self.num_classes, cfg=cfg_model, **cfg.Model).to(self.device)
        self.SMT(model=model_new)


@hydra.main(version_base=None, config_path='./Configs', config_name='defaults')
def run(cfg: omegaconf.DictConfig):
    from itertools import permutations
    TL_Task_list = list(permutations(cfg.Dataset.TL_list, 2))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    import time
    train_time = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
    
    seed_list = list(range(2020, 2025))
    seed_list = list(range(2025, 2026))
    for seed_run in seed_list:
        for TL_Task in TL_Task_list:
            with open_dict(cfg):
                cfg.train_time = train_time
                cfg.Dataset.TL_Task = TL_Task
                if cfg.Dataset.data_name == 'Gear':
                    cfg.Opt.lr_src = 1e-3
                elif cfg.Dataset.data_name == 'CWRU':
                    cfg.Opt.lr_src = 1e-3
                elif cfg.Dataset.data_name == 'PU':
                    cfg.Opt.lr_src = 1e-3
                cfg.Opt.lr_scheduler = 'designed'
                cfg.Opt.weight_decay_src = cfg.Opt.lr_src / 10
                cfg.Dataset.input_kind = 'fft'
                cfg.Model.bottleneck_num = 128
                cfg.seed_run = seed_run
                cfg.wandb.setup.name = str(TL_Task) + 'Task_' + str(cfg.seed_run)
                cfg.PR = 0.2 # prune ratio (PR) value  for source model pruning
                #cfg.prune_percent = 0 # no source model pruning
                cfg.src_epoch = 50

                cfg_info_base = cfg.Dataset.input_kind + '_' + cfg.Model.model_name + '_lr_' + str(cfg.Opt.lr_src) + '_PR_' + str(cfg.PR)
                group_name = train_time + cfg_info_base
                cfg.wandb.setup.group = group_name
                cfg.wandb.setup.project = "TTA_Src" + '_' + cfg.Dataset.data_name

                cfg.save_models = True
                cfg.load_models = True

                wandb_cfg = omegaconf.OmegaConf.to_container(
                    cfg, resolve=True, throw_on_missing=True
                )
                run = wandb.init(config=wandb_cfg, **cfg.wandb.setup)
                with run:
                    Src = Source_Trainer(cfg, run)
                    Src.setup()
                    print(f"--------{cfg.Dataset.data_name} Dataset:{TL_Task} TL Task---------")
                    print(f"the input kind is {cfg.Dataset.input_kind},"
                          f"the model kind is {cfg.Model.model_name}")
                    cfg_dict = omegaconf.OmegaConf.to_container(cfg)
                    print(f"the opt infor is {cfg_dict['Opt']}")
                    print(f"the data infor is {cfg_dict['Dataset']}")
                    cfg.process_wandb = True
                    if cfg.PR == 0:
                        Src.SMT()
                    elif cfg.PR > 0 and cfg.PR < 1:
                        Src.SMP()
                    else:
                        raise ValueError('PR should be in [0,1]')

            print(f"--------{cfg.Dataset.data_name} Dataset:{TL_Task} TL Task---------")
            print(f"the input kind is {cfg.Dataset.input_kind},"
                  f"the model kind is {cfg.Model.model_name}")
            cfg_dict = omegaconf.OmegaConf.to_container(cfg)
            print(f"the opt infor is {cfg_dict['Opt']}")
            print(f"the data infor is {cfg_dict['Dataset']}")



if __name__ == '__main__':
    run()
    



