from yolox.exp import Exp
import os

class Exp(Exp):
    def __init__(self):
        super(Exp, self).__init__()
        self.data_num_workers = 4
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.random_size = (14, 26)
        self.data_dir = "Path to your dataset root"
        self.train_ann = "Path to your .json train annotations"
        self.val_ann = "Path to your .json val annotations"
        self.num_classes = 1

        self.warmup_epochs = 5
        self.max_epoch = 300
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        # This is all data augmentation parameters that we can tweak to get better training results. 
        # Usually to detect smaller objects using mosaic, scale and mixup is good.
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2)
        self.mosaic_scale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True