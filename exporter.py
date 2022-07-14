import time
from models.sgdepth import SGDepth
import torch
from arguments import InferenceEvaluationArguments
import cv2
import os
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import glob as glob
from torchinfo import summary
import struct 

outputs = {} 

def hook(module,input,output):
    outputs[module] = output

def bin_write(f,data):
    data = data.flatten()
    fmt = 'f'*len(data)
    bin = struct.pack(fmt,*data)
    f.write(bin)

class Exporter:
    
    def __init__(self):
        self.model_path = opt.model_path
        self.image_dir = opt.image_path
        self.image_path = opt.image_path
        self.num_classes = 20
        self.depth_min = opt.model_depth_min
        self.depth_max = opt.model_depth_max
        self.output_path = opt.output_path
        self.output_format = opt.output_format
        self.all_time = []
        self.labels = (('CLS_ROAD', (128, 64, 128)),
                       ('CLS_SIDEWALK', (244, 35, 232)),
                       ('CLS_BUILDING', (70, 70, 70)),
                       ('CLS_WALL', (102, 102, 156)),
                       ('CLS_FENCE', (190, 153, 153)),
                       ('CLS_POLE', (153, 153, 153)),
                       ('CLS_TRLIGHT', (250, 170, 30)),
                       ('CLS_TRSIGN', (220, 220, 0)),
                       ('CLS_VEGT', (107, 142, 35)),
                       ('CLS_TERR', (152, 251, 152)),
                       ('CLS_SKY', (70, 130, 180)),
                       ('CLS_PERSON', (220, 20, 60)),
                       ('CLS_RIDER', (255, 0, 0)),
                       ('CLS_CAR', (0, 0, 142)),
                       ('CLS_TRUCK', (0, 0, 70)),
                       ('CLS_BUS', (0, 60, 100)),
                       ('CLS_TRAIN', (0, 80, 100)),
                       ('CLS_MCYCLE', (0, 0, 230)),
                       ('CLS_BCYCLE', (119, 11, 32)),
                       )
    
    def init_model(self):
        print("Init Model...")
        sgdepth = SGDepth

        with torch.no_grad():
            # init 'empty' model
            self.model = sgdepth(
                opt.model_split_pos, opt.model_num_layers, opt.train_depth_grad_scale,
                opt.train_segmentation_grad_scale,
                # opt.train_domain_grad_scale,
                opt.train_weights_init, opt.model_depth_resolutions, opt.model_num_layers_pose,
                # opt.model_num_domains,
                # opt.train_loss_weighting_strategy,
                # opt.train_grad_scale_weighting_strategy,
                # opt.train_gradnorm_alpha,
                # opt.train_uncertainty_eta_depth,
                # opt.train_uncertainty_eta_seg,
                # opt.model_shared_encoder_batchnorm_momentum
            )

            # load weights (copied from state manager)
            state = self.model.state_dict()
            to_load = torch.load(self.model_path)
            for (k, v) in to_load.items():
                if k not in state:
                    print(f"    - WARNING: Model file contains unknown key {k} ({list(v.shape)})")

            for (k, v) in state.items():
                if k not in to_load:
                    print(f"    - WARNING: Model file does not contain key {k} ({list(v.shape)})")

                else:
                    state[k] = to_load[k]

            self.model.load_state_dict(state)
            self.model = self.model.eval().cuda()  

    def exporter(self):
        self.init_model()
        for n,m in self.model.named_modules():
            m.register_forward_hook(hook)

        f = None
        for n,m in self.model.named_modules():
            t = '-'.join(n.split('.'))

            if not('of Conv2d' in str(m.type) or 'of BatchNorm2d' in str(m.type)):
                continue

            if 'of Conv2d' in str(m.type):
                file_name = "tkdnn_bin/layers/" + t + ".bin"
                print("open file: ", file_name)
                f = open(file_name, mode='wb')

                w = np.array([])
                b = np.array([])
                if 'weight' in m._parameters and m._parameters['weight'] is not None:
                    w = m._parameters['weight'].cpu().data.numpy()
                    w = np.array(w, dtype=np.float32)
                    print("    weights shape:", np.shape(w))

                if 'bias' in m._parameters and m._parameters['bias'] is not None:
                    b = m._parameters['bias'].cpu().data.numpy()
                    b = np.array(b, dtype=np.float32)
                    print("    bias shape:", np.shape(b))
                
                bin_write(f, w)
                bias_shape = w.shape[0]
                if b.size > 0:
                    bin_write(f, b)
                f.close()
                print("close file")
                f = None
            if 'of BatchNorm2d' in str(m.type):
                file_name = "tkdnn_bin/layers/" + t + ".bin"
                print("open file: ",file_name)
                f = open(file_name,mode='wb')
                b = m._parameters['bias'].cpu().data.numpy()
                b = np.array(b, dtype=np.float32)
                s = m._parameters['weight'].cpu().data.numpy()
                s = np.array(s, dtype=np.float32)
                rm = m.running_mean.cpu().data.numpy()
                rm = np.array(rm, dtype=np.float32)
                rv = m.running_var.cpu().data.numpy()
                rv = np.array(rv, dtype=np.float32)
                bin_write(f, b)
                bin_write(f, s)
                bin_write(f, rm)
                bin_write(f, rv)

                print("    b shape:", np.shape(b))
                print("    s shape:", np.shape(s))
                print("    rm shape:", np.shape(rm))
                print("    rv shape:", np.shape(rv))

                f.close()
                print("close file")
        print(self.model)
        
if __name__ == "__main__":
    opt = InferenceEvaluationArguments().parse()

    export = Exporter()
    export.exporter()
