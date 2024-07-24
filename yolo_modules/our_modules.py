
import contextlib
from ultralytics.nn.modules import *
from torch import nn
import contextlib
from copy import deepcopy
from pathlib import Path

import thop
import torch
import torch.nn as nn

from ultralytics.nn.modules import (C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, Classify,
                                    Concat, Conv, ConvTranspose, Detect, DWConv, DWConvTranspose2d, Focus,
                                    GhostBottleneck, GhostConv, Segment)
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, RANK, colorstr, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml
from ultralytics.utils.torch_utils import (fuse_conv_and_bn, fuse_deconv_and_bn, initialize_weights,
                                                intersect_dicts, make_divisible, model_info, scale_img, time_sync)

# class DetectionModel(BaseModel):
#     # YOLOv8 detection model
#     def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):  # model, input channels, number of classes
#         super().__init__()
#         self.yaml = cfg if isinstance(cfg, dict) else yaml_load(check_yaml(cfg), append_filename=True)  # cfg dict

#         # Define model
#         ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
#         if nc and nc != self.yaml['nc']:
#             LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
#             self.yaml['nc'] = nc  # override yaml value
#         self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
#         self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
#         self.inplace = self.yaml.get('inplace', True)

#         # Build strides
#         m = self.model[-1]  # Detect()
#         if isinstance(m, (Detect, Segment)):
#             s = 256  # 2x min stride
#             m.inplace = self.inplace
#             forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
#             m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
#             self.stride = m.stride
#             m.bias_init()  # only run once

#         # Init weights, biases
#         initialize_weights(self)
#         if verbose:
#             self.info()
#             LOGGER.info('')

#     def forward(self, x, augment=False, profile=False, visualize=False):
#         if augment:
#             return self._forward_augment(x)  # augmented inference, None
#         return self._forward_once(x, profile, visualize)  # single-scale inference, train

#     def _forward_augment(self, x):
#         img_size = x.shape[-2:]  # height, width
#         s = [1, 0.83, 0.67]  # scales
#         f = [None, 3, None]  # flips (2-ud, 3-lr)
#         y = []  # outputs
#         for si, fi in zip(s, f):
#             xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
#             yi = self._forward_once(xi)[0]  # forward
#             # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
#             yi = self._descale_pred(yi, fi, si, img_size)
#             y.append(yi)
#         y = self._clip_augmented(y)  # clip augmented tails
#         return torch.cat(y, -1), None  # augmented inference, train

#     @staticmethod
#     def _descale_pred(p, flips, scale, img_size, dim=1):
#         # de-scale predictions following augmented inference (inverse operation)
#         p[:, :4] /= scale  # de-scale
#         x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
#         if flips == 2:
#             y = img_size[0] - y  # de-flip ud
#         elif flips == 3:
#             x = img_size[1] - x  # de-flip lr
#         return torch.cat((x, y, wh, cls), dim)

#     def _clip_augmented(self, y):
#         # Clip YOLOv5 augmented inference tails
#         nl = self.model[-1].nl  # number of detection layers (P3-P5)
#         g = sum(4 ** x for x in range(nl))  # grid points
#         e = 1  # exclude layer count
#         i = (y[0].shape[-1] // g) * sum(4 ** x for x in range(e))  # indices
#         y[0] = y[0][..., :-i]  # large
#         i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
#         y[-1] = y[-1][..., i:]  # small
#         return y

#     def load(self, weights, verbose=True):
#         csd = weights.float().state_dict()  # checkpoint state_dict as FP32
#         csd = intersect_dicts(csd, self.state_dict())  # intersect
#         self.load_state_dict(csd, strict=False)  # load
#         if verbose and RANK == -1:
#             LOGGER.info(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights')


def import_str(class_):
    name = class_.__name__
    cmp_name  = str(class_)[8: -3]
    parent_packages = cmp_name.split(".")
    parent_package = ".".join(parent_packages[:-1])
    
    return f"from {parent_package} import {name}"
    
    
def str_list(data_list):
    return [f"\"{data}\"" if isinstance(data, str) else str(data) for data in data_list]

def parse_model_from(model_class_name, yaml_file, ch=3):
    yaml_dict = load_yaml_(yaml_file)
    model, save_list, a, b, c = parse_model(yaml_dict, ch)
    forward_codes = forward_once(model)
    python_codes = expand_forward(model_class_name, a, b, forward_codes)
    return  model, save_list, python_codes

def load_yaml_(file_path):
    yaml_dict = yaml_load(file_path)
    return yaml_dict

def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    # Parse a YOLO model.yaml dictionary
    import_str_set = set()
    import_str_set.add("import torch.nn as nn\n")
    import_str_set.add("from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, RANK, colorstr")
    import_str_set.add("from ultralytics.nn.tasks import *")
    concat_params = []
    sub_modules = []
    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    nc, gd, gw, act = d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            # TODO: re-implement with eval() removal if possible
            # args[j] = (locals()[a] if a in locals() else ast.literal_eval(a)) if isinstance(a, str) else a
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
                Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(args[2] * gw, 8)
        else:
            c2 = ch[f]

        if n > 1:
            squ = []
            init_list = []
            for _ in range(n) :
                squ.append(*(m(*args) ))
                args_str = ",".join(str_list(args))
                init_str = f"{m.__name__}({args_str})"
                init_list.append(init_str)
                import_str_set.add(import_str(m))                
                
                instance_list = ",\n".join(init_list)
            m_ = nn.Sequential(squ) 
            
            sub_module = f"nn.Sequential({instance_list})"
            import_str_set.add("from torch import nn")
        else:
            m_ = m(*args)  # module
            args_str = ",".join(str_list(args))
            init_str = f"{m.__name__}({args_str})"
            import_str_set.add(import_str(m))
            
            sub_module = init_str
        
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        
        sub_modules.append(sub_module)
        
        if i == 0:
            ch = []
        ch.append(c2)
    for import_str_ in import_str_set:
        print(import_str_)
    sub_modules_str = ",\n\t\t\t".join(sub_modules)
    init_str = f"\t\tself.model = nn.Sequential([{sub_modules_str}])"
    print(init_str)
    concat_params.append((i, f, t))
    
    
    return nn.Sequential(*layers), sorted(save), init_str, import_str_set, concat_params


def construct_import_set(import_str_set):
        pass

# def gen_py_codes(instance):
#     import inspect
#     print(inspect.getsource(instance))

def forward_once(model):
    codes = []
    # y, dt = [], []  # outputs
    codes.append("y, dt = [], []")
    for idx, m in enumerate(model):
        codes.append(f"\"\"\" {m.type} \"\"\"")
        codes.append(f"m = self.model[{idx}]")
        if m.f != -1:
            # x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if isinstance(m.f, int):
                # x = y[m.f]
                codes.append(f"x = y[{m.f}]")
            else:
                # x = [x if j == -1 else y[j] for j in m.f]
                codes.append(f"x =  [x if j == -1 else y[j] for j in {m.f}]")
                
        codes.append("if profile:\n\t\t\t\tself._profile_one_layer(m, x, dt)")
        codes.append("x = m(x)")
        codes.append("y.append(x if m.i in self.save else None)  # save output")
        # codes.append("""if visualize:\n\t\t\t\tLOGGER.info('visualize feature not yet supported')\n\t\t# TODO: feature_visualization(x, m.type, m.i, save_dir=visualize)""" )
        codes.append("""
		if visualize:
			feature_visualization(x, m.type, m.i, save_dir=visualize)
	
	""")
    codes.append("return x")
    
    code_str = ""
    for code_line in codes:
        code_str +=  "\t\t" + code_line + "\n"
    return code_str
    

def expand_forward(class_name, init_str, import_strs, forward_str, parent_class="DetectionModel"):
    
    import_head = ""
    for i in import_strs:
        import_head += i + "\n"
    
    print(import_head)
    
    python_code = f"""
{import_head}
class {class_name}({parent_class}):
\tdef __init__(self) -> None:
\t\tsuper().__init__()
{init_str}
            
\tdef _predict_once(self, x, profile=False, visualize=False):
{forward_str}
    """
    print(python_code)
    return python_code
    
    
if __name__ == "__main__":
    a, b, c = parse_model_from("YoloModel", "/nasdata/private/zwlu/Now/ai_trainer/yolo_config/default_yolov8m_20_classes.yaml")
        
    with open("/nasdata/private/zwlu/Now/ai_trainer/yolo_modules/test_model.py", "w") as model_test:
        model_test.write(c)
# end main