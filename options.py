import argparse
import os
import torch
import random
import torch.backends.cudnn as cudnn
from utils import mkdir_p

import datetime


def _safe(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s))


class Options(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):

        # self.parser = argparse.ArgumentParser( description='STdeblur training.' )
        # # Training
        self.parser.add_argument(
            "--run_root",
            default="runs",
            type=str,
            help="root directory to create unique run folders",
        )
        self.parser.add_argument(
            "--epochs",
            default=1000,
            type=int,
            metavar="N",
            help="number of total epochs to run",
        )
        self.parser.add_argument(
            "--start-epoch",
            default=0,
            type=int,
            metavar="N",
            help="manual epoch number (useful on restarts)",
        )
        self.parser.add_argument(
            "--batchSize", default=8, type=int, metavar="N", help="input batch size"
        )
        self.parser.add_argument(
            "--lr",
            "--learning-rate",
            default=1e-7,
            type=float,
            metavar="LR",
            help="initial learning rate",
        )
        self.parser.add_argument(
            "--momentum", default=0.9, type=float, metavar="M", help="momentum"
        )
        self.parser.add_argument(
            "--weight-decay",
            "--wd",
            default=0,
            type=float,
            metavar="W",
            help="weight decay (default: 1e-4)",
        )
        self.parser.add_argument(
            "--schedule",
            type=int,
            nargs="+",
            default=20,
            help="Decrease learning rate at these epochs.",
        )
        self.parser.add_argument(
            "--gamma",
            type=float,
            default=0.9,
            help="LR is mult-\
                                 iplied by gamma on schedule.",
        )
        # GPU
        self.parser.add_argument(
            "--gpu_ids",
            type=str,
            default="1,4,5,7",
            help="gpu ids: \
                                e.g. 0  0,1,2, 0,2. use -1 for CPU",
        )
        self.parser.add_argument("--manualSeed", type=int, help="manual seed")
        # self.parser.add_argument('--use_cuda', type=bool, default=False, help='use_cuda bool')

        # Dataset
        self.parser.add_argument(
            "--dataroot",
            type=str,
            default="./ICIPDataset",
            help="path to\
                                images (should have subfolders train/blurred, train/sharp,\
                                val/blurred, val/sharp, test/blurred, test/sharp etc)",
        )

        self.parser.add_argument(
            "--train_classes",
            type=str,
            default="appbox,baisui,bingho,cokele,haitun,jianjo,pacup1,pacup2,zhijin",
            help="comma-separated class names used for TRAIN split",
        )
        self.parser.add_argument(
            "--test_classes",
            type=str,
            default="cesbon,nongf1,songsu",
            help="comma-separated class names used for TEST split",
        )

        self.parser.add_argument(
            "--phase",
            type=str,
            default="train",
            help="train, val,\
                                test, etc",
        )
        self.parser.add_argument(
            "--cropWidth",
            type=int,
            default=112,
            help="Crop to\
                                this width",
        )
        self.parser.add_argument(
            "--cropHeight",
            type=int,
            default=112,
            help="Crop to\
                                this height",
        )
        self.parser.add_argument(
            "-j",
            "--workers",
            default=4,
            type=int,
            metavar="N",
            help="number of data loading workers (default: 4)",
        )
        # Checkpoints
        self.parser.add_argument(
            "-c",
            "--checkpoint",
            default="checkpoint_sgd",
            type=str,
            metavar="PATH",
            help="path to save checkpoint (default: checkpoint)",
        )
        self.parser.add_argument(
            "--resume",
            default="",
            type=str,
            metavar="PATH",
            help="path to latest checkpoint (default: none)",
        )  # ./checkpoint/model_best.pth.tar
        self.parser.add_argument(
            "--name",
            type=str,
            default="experiment_name",
            help="name of\
                                the experiment. It decides where to store samples and models",
        )
        # miscs
        self.parser.add_argument(
            "-e",
            "--evaluate",
            dest="evaluate",
            action="store_true",
            help="evaluate model on validation set",
        )
        self.parser.add_argument(
            "--model_arch", type=str, default="C3D", help="The model arch you selected"
        )

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        def _split_csv(s: str):
            return [x.strip() for x in s.split(",") if x.strip()]

        self.opt.train_classes = _split_csv(self.opt.train_classes)
        self.opt.test_classes = _split_csv(self.opt.test_classes)

        # GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = self.opt.gpu_ids
        self.opt.use_cuda = torch.cuda.is_available()
        str_ids = self.opt.gpu_ids.split(",")
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # Random seed
        if self.opt.manualSeed is None:
            self.opt.manualSeed = random.randint(1, 10000)
        random.seed(self.opt.manualSeed)
        torch.manual_seed(self.opt.manualSeed)
        if self.opt.use_cuda:
            torch.cuda.manual_seed_all(self.opt.manualSeed)
            cudnn.benchmark = True
            cudnn.enabled = True

        args = vars(self.opt)
        print("------------ Options -------------")
        for k, v in sorted(args.items()):
            print("%s: %s" % (str(k), str(v)))
        print("-------------- End ----------------")

        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]  # ミリ秒まで
        exp = _safe(self.opt.name)
        tag = _safe(f"{self.opt.model_arch}_bs{self.opt.batchSize}_lr{self.opt.lr}")

        # run_dir = runs/20260203-083033-123_expname_C3D_bs8_lr0.0001
        run_dir = os.path.join(self.opt.run_root, f"{ts}_{exp}_{tag}")
        mkdir_p(run_dir)

        # これ以降 “すべての保存先” を run_dir に統一
        self.opt.run_dir = run_dir
        self.opt.checkpoint = os.path.join(run_dir, "checkpoints")
        self.opt.results_dir = os.path.join(run_dir, "results")
        mkdir_p(self.opt.checkpoint)
        mkdir_p(self.opt.results_dir)

        # opt.txt も run_dir に出す（衝突しない）
        file_name = os.path.join(run_dir, "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write("------------ Options -------------\n")
            for k, v in sorted(args.items()):
                opt_file.write("%s: %s\n" % (str(k), str(v)))
            opt_file.write("-------------- End ----------------\n")

        print("RUN_DIR:", run_dir)
        return self.opt

        # expr_dir = os.path.join(self.opt.checkpoint, self.opt.name)
        # mkdir_p(expr_dir)
        # file_name = os.path.join(expr_dir, "opt.txt")
        # with open(file_name, "wt") as opt_file:
        #    opt_file.write("------------ Options -------------\n")
        #    for k, v in sorted(args.items()):
        #        opt_file.write("%s: %s\n" % (str(k), str(v)))
        #    opt_file.write("-------------- End ----------------\n")
        # return self.opt
