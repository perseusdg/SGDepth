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
    def load_image(self):
        

        self.image = Image.open("dog.jpg")  # open PIL image
        self.image_o_width, self.image_o_height = self.image.size

        resize = transforms.Resize(
            (opt.inference_resize_height, opt.inference_resize_width))
        image = resize(self.image)  # resize to argument size

        #center_crop = transforms.CenterCrop((opt.inference_crop_height, opt.inference_crop_width))
        #image = center_crop(image)  # crop to input size

        to_tensor = transforms.ToTensor()  # transform to tensor

        self.input_image = to_tensor(image)  # save tensor image to self.input_image for saving later
        image = self.normalize(self.input_image)

        image = image.unsqueeze(0).float().cuda()
        image2 = image
        input_image_array = np.array(image2.cpu().detach().numpy(),dtype=np.float32)
        input_image_array.tofile("tkdnn_bin/inputs/input.bin",format="f")

        # simulate structure of batch:
        image_dict = {('color_aug', 0, 0): image}  # dict
        image_dict[('color', 0, 0)] = image
        image_dict['domain'] = ['cityscapes_val_seg', ]
        image_dict['purposes'] = [['segmentation', ], ['depth', ]]
        image_dict['num_classes'] = torch.tensor([self.num_classes])
        image_dict['domain_idx'] = torch.tensor(0)
        self.batch = (image_dict,)  # batch tuple

    def normalize(self, tensor):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        normalize = transforms.Normalize(mean, std)
        tensor = normalize(tensor)

        return tensor       

    def exporter(self):
        if not os.path.exists('tkdnn_bin'):
            os.makedirs('tkdnn_bin')    
        if not os.path.exists('kdnn_bin/debug'):
            os.makedirs('tkdnn_bin/debug')
        if not os.path.exists('tkdnn_bin/layers'):
            os.makedirs('tkdnn_bin/layers')
        if not os.path.exists('tkdnn_bin/outputs'):
            os.makedirs('tkdnn_bin/outputs')
        if not os.path.exists('tkdnn_bin/inputs'):
            os.makedirs('tkdnn_bin/inputs')

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

                f = open(file_name, mode='wb')

                w = np.array([])
                b = np.array([])
                if 'weight' in m._parameters and m._parameters['weight'] is not None:
                    w = m._parameters['weight'].cpu().data.numpy()
                    w = np.array(w, dtype=np.float32)


                if 'bias' in m._parameters and m._parameters['bias'] is not None:
                    b = m._parameters['bias'].cpu().data.numpy()
                    b = np.array(b, dtype=np.float32)

                
                bin_write(f, w)
                bias_shape = w.shape[0]
                if b.size > 0:
                    bin_write(f, b)
                else:
               	    bin_write(f,np.zeros(bias_shape))
               	    bias_shape=0
                f.close()

                f = None
            if 'of BatchNorm2d' in str(m.type):
                file_name = "tkdnn_bin/layers/" + t + ".bin"

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

               

                f.close()


        self.load_image()
        output = self.model(self.batch)
        for i in range(0,4):
            disps_pred_temp = output[0]["disp",i]
            print(disps_pred_temp.size())
            disps_pred_temp_np_array = np.array(disps_pred_temp.cpu().detach().numpy(),dtype=np.float32)
            output_file_name = "tkdnn_bin/outputs/depth_decoder_output_" + str(i) + ".bin"
            disps_pred_temp_np_array.tofile(output_file_name,format="f")
        segs_pred = output[0]['segmentation_logits', 0] 
        segs_pred = segs_pred.exp()
        print(segs_pred.size())
        seg_pred_temp_np_array = np.array(segs_pred.cpu().detach().numpy(),dtype=np.float32)
        output_file_name = "tkdnn_bin/outputs/seg_decoder_output.bin"
        seg_pred_temp_np_array.tofile(output_file_name,format="f")
        print(self.model)
        

        
if __name__ == "__main__":
    opt = InferenceEvaluationArguments().parse()

    export = Exporter()
    export.exporter()
