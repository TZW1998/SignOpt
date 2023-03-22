import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

from inspect import signature
from collections import namedtuple
import time
import torch
from torch import nn
import numpy as np
import torchvision
import pandas as pd

torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}
def res_block(c_in, c_out, stride, **kw):
    block = {
        'bn1': batch_norm(c_in, **kw),
        'relu1': nn.ReLU(True),
        'branch': {
            'conv1': nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
            'bn2': batch_norm(c_out, **kw),
            'relu2': nn.ReLU(True),
            'conv2': nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False),
        }
    }
    projection = (stride != 1) or (c_in != c_out)    
    if projection:
        block['conv3'] = (nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, padding=0, bias=False), [rel_path('relu1')])
    block['add'] =  (Add(), [(rel_path('conv3') if projection else rel_path('relu1')), rel_path('branch', 'conv2')])
    return block

def DAWN_net(c=64, block=res_block, prep_bn_relu=False, concat_pool=True, **kw):    
    if isinstance(c, int):
        c = [c, 2*c, 4*c, 4*c]
        
    classifier_pool = {
        'in': Identity(),
        'maxpool': nn.MaxPool2d(4),
        'avgpool': (nn.AvgPool2d(4), [rel_path('in')]),
        'concat': (Concat(), [rel_path('maxpool'), rel_path('avgpool')]),
    } if concat_pool else {'pool': nn.MaxPool2d(4)}
    
    return {
        'prep': union({'conv': nn.Conv2d(3, c[0], kernel_size=3, stride=1, padding=1, bias=False)},
                      {'bn': batch_norm(c[0], **kw), 'relu': nn.ReLU(True)} if prep_bn_relu else {}),
        'layer1': {
            'block0': block(c[0], c[0], 1, **kw),
            'block1': block(c[0], c[0], 1, **kw),
        },
        'layer2': {
            'block0': block(c[0], c[1], 2, **kw),
            'block1': block(c[1], c[1], 1, **kw),
        },
        'layer3': {
            'block0': block(c[1], c[2], 2, **kw),
            'block1': block(c[2], c[2], 1, **kw),
        },
        'layer4': {
            'block0': block(c[2], c[3], 2, **kw),
            'block1': block(c[3], c[3], 1, **kw),
        },
        'classifier': union(classifier_pool, {
            'flatten': Flatten(),
            'linear': nn.Linear(2*c[3] if concat_pool else c[3], 10, bias=True),
            'logits': Identity()
        })
    }


def conv_bn(c_in, c_out, bn_weight_init=1.0, **kw):
    return {
        'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False), 
        'bn': batch_norm(c_out, bn_weight_init=bn_weight_init, **kw), 
        'relu': nn.ReLU(True)
    }

def basic_net(channels, weight,  pool, **kw):
    return {
        'prep': conv_bn(3, channels['prep'], **kw),
        'layer1': dict(conv_bn(channels['prep'], channels['layer1'], **kw), pool=pool),
        'layer2': dict(conv_bn(channels['layer1'], channels['layer2'], **kw), pool=pool),
        'layer3': dict(conv_bn(channels['layer2'], channels['layer3'], **kw), pool=pool),
        'classifier': {
            'pool': nn.MaxPool2d(4),
            'flatten': Flatten(),
            'linear': nn.Linear(channels['layer3'], 10, bias=False),
            'logits': Mul(weight),
        }
    }

def fastnet(channels=None, weight=0.125, pool=nn.MaxPool2d(2), extra_layers=(), res_layers=('layer1', 'layer3'), **kw):
    channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
    residual = lambda c, **kw: {'in': Identity(), 'res1': conv_bn(c, c, **kw), 'res2': conv_bn(c, c, **kw), 
                                'add': (Add(), [rel_path('in'), rel_path('res2', 'relu')])}
    n = basic_net(channels, weight, pool, **kw)
    for layer in res_layers:
        n[layer]['residual'] = residual(channels[layer], **kw)
    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer], **kw)     

#     losses = {
#     'loss':  (nn.CrossEntropyLoss(size_average=False), [('classifier','logits'), ('target',)]),
#     'correct': (Correct(), [('classifier','logits'), ('target',)]),
# }  
    return TorchGraph(n)

def fastnet2(channels=None, weight=0.125, pool=nn.MaxPool2d(2), extra_layers=(), res_layers=('layer1', 'layer3'), **kw):
    channels = channels or {'prep': 128, 'layer1': 256, 'layer2': 512, 'layer3': 1024}
    residual = lambda c, **kw: {'in': Identity(), 'res1': conv_bn(c, c, **kw), 'res2': conv_bn(c, c, **kw), 
                                'add': (Add(), [rel_path('in'), rel_path('res2', 'relu')])}
    n = basic_net(channels, weight, pool, **kw)
    for layer in res_layers:
        n[layer]['residual'] = residual(channels[layer], **kw)
    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer], **kw)     

#     losses = {
#     'loss':  (nn.CrossEntropyLoss(size_average=False), [('classifier','logits'), ('target',)]),
#     'correct': (Correct(), [('classifier','logits'), ('target',)]),
# }  
    return TorchGraph(n)


#####################
# utils
#####################

class Timer():
    def __init__(self):
        self.times = [time.time()]
        self.total_time = 0.0

    def __call__(self, include_in_total=True):
        self.times.append(time.time())
        dt = self.times[-1] - self.times[-2]
        if include_in_total:
            self.total_time += dt
        return dt
    
localtime = lambda: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

def warmup_cudnn(model, batch_size):
    #run forward and backward pass of the model on a batch of random inputs
    #to allow benchmarking of cudnn kernels 
    batch = {
        'input': torch.Tensor(np.random.rand(batch_size,3,32,32)).cuda().half(), 
        'target': torch.LongTensor(np.random.randint(0,10,batch_size)).cuda()
    }
    model.train(True)
    o = model(batch)
    o['loss'].backward()
    model.zero_grad()
    torch.cuda.synchronize()

class TableLogger():
    def append(self, output):
        if not hasattr(self, 'keys'):
            self.keys = output.keys()
            print(*(f'{k:>12s}' for k in self.keys))
        filtered = [output[k] for k in self.keys]
        print(*(f'{v:12.4f}' if isinstance(v, np.float) else f'{v:12}' for v in filtered))

#####################
## data preprocessing
#####################

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

#####################
## data augmentation
#####################

class Crop(namedtuple('Crop', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        return x[:,y0:y0+self.h,x0:x0+self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)}
    
    def output_shape(self, x_shape):
        C, H, W = x_shape
        return (C, self.h, self.w)
    
class FlipLR(namedtuple('FlipLR', ())):
    def __call__(self, x, choice):
        return x[:, :, ::-1].copy() if choice else x 
        
    def options(self, x_shape):
        return {'choice': [True, False]}

class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:,y0:y0+self.h,x0:x0+self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)} 
    
    
class Transform():
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None
        
    def __len__(self):
        return len(self.dataset)
           
    def __getitem__(self, index):
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k,v) in choices.items()}
            data = f(data, **args)
        return data, labels
    
    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(t, 'output_shape') else x_shape
            self.choices.append({k:np.random.choice(v, size=N) for (k,v) in options.items()})

#####################
## data loading
#####################

class Batches():
    def __init__(self, dataset, batch_size, shuffle, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )
    
    def __iter__(self):  
        return ({'input': x.to(device).half(), 'target': y.to(device).long()} for (x,y) in self.dataloader)
    
    def __len__(self): 
        return len(self.dataloader)


#####################
## torch stuff
#####################

class Identity(nn.Module):
    def forward(self, x): return x
    
class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x): 
        return x*self.weight
    
class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

class Add(nn.Module):
    def forward(self, x, y): return x + y 
    
class Concat(nn.Module):
    def forward(self, *xs): return torch.cat(xs, 1)
    
class Correct(nn.Module):
    def forward(self, classifier, target):
        return classifier.max(dim = 1)[1] == target

def batch_norm(num_channels, bn_bias_init=None, bn_bias_freeze=False, bn_weight_init=None, bn_weight_freeze=False):
    m = nn.BatchNorm2d(num_channels)
    if bn_bias_init is not None:
        m.bias.data.fill_(bn_bias_init)
    if bn_bias_freeze:
        m.bias.requires_grad = False
    if bn_weight_init is not None:
        m.weight.data.fill_(bn_weight_init)
    if bn_weight_freeze:
        m.weight.requires_grad = False
        
    return m

def to_numpy(x):
    return x.detach().cpu().numpy()  

#####################
## dict utils
#####################



def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict): yield from path_iter(val, (*pfx, name))
        else: yield ((*pfx, name), val)  

#####################
## graph building
#####################

sep='_'
RelativePath = namedtuple('RelativePath', ('parts'))
rel_path = lambda *parts: RelativePath(parts)

def build_graph(net):
    net = dict(path_iter(net)) 
    default_inputs = [[('input',)]]+[[k] for k in net.keys()]
    with_default_inputs = lambda vals: (val if isinstance(val, tuple) else (val, default_inputs[idx]) for idx,val in enumerate(vals))
    parts = lambda path, pfx: tuple(pfx) + path.parts if isinstance(path, RelativePath) else (path,) if isinstance(path, str) else path
    return {sep.join((*pfx, name)): (val, [sep.join(parts(x, pfx)) for x in inputs]) for (*pfx, name), (val, inputs) in zip(net.keys(), with_default_inputs(net.values()))}
    
class TorchGraph(nn.Module):
    def __init__(self, net):
        self.graph = build_graph(net)
        super().__init__()
        for n, (v, _) in self.graph.items(): 
            setattr(self, n, v)

    def forward(self, inputs):
        self.cache = {'input': inputs}
        for n, (_, i) in self.graph.items():
            self.cache[n] = getattr(self, n)(*[self.cache[x] for x in i])
        return self.cache["classifier_logits"]
    
    def half(self):
        for module in self.children():
            if type(module) is not nn.BatchNorm2d:
                module.half()    
        return self

#####################
## training utils
#####################

class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]

trainable_params = lambda model:filter(lambda p: p.requires_grad, model.parameters())

def nesterov(params, momentum, weight_decay=None):
    return torch.optim.SGD(params, lr=0.0, momentum=momentum, weight_decay=weight_decay, nesterov=True)

concat = lambda xs: np.array(xs) if xs[0].shape is () else np.concatenate(xs)

def set_opt_params(optimizer, params):
    for k, v in params.items():
        optimizer.param_groups[0][k] = v
    return optimizer

def update(model, optimizer):
    assert model.training
    model.cache['loss'].backward()
    optimizer.step()
    model.zero_grad()
    
def collect(stats, output):
    for k,v in stats.items():
        v.append(to_numpy(output[k]))

def train_epoch(model, batches, optimizer, lrs, stats):
    model.train(True)   
    for lr, batch in zip(lrs, batches):
        collect(stats, model(batch))
        update(model, set_opt_params(optimizer, {'lr': lr}))
    return stats

def test_epoch(model, batches, stats):
    model.train(False)
    for batch in batches:
        collect(stats, model(batch))
    return stats

sum_ = lambda xs: np.sum(concat(xs), dtype=np.float)

def train(model, lr_schedule, optimizer, train_set, test_set, batch_size=512, 
          loggers=(), test_time_in_total=True, num_workers=0, drop_last=False, timer=None):  
    t = timer or Timer()
    train_batches = Batches(train_set, batch_size, shuffle=True, num_workers=num_workers, drop_last=drop_last)
    test_batches = Batches(test_set, batch_size, shuffle=False, num_workers=num_workers)
    N_train, N_test = len(train_set), len(test_set)
    if drop_last: N_train -= (N_train % batch_size)

    for epoch in range(lr_schedule.knots[-1]):
        train_batches.dataset.set_random_choices() 
        lrs = (lr_schedule(x)/batch_size for x in np.arange(epoch, epoch+1, 1/len(train_batches)))
        train_stats, train_time = train_epoch(model, train_batches, optimizer, lrs, {'loss': [], 'correct': []}), t()
        test_stats, test_time = test_epoch(model, test_batches, {'loss': [], 'correct': []}), t(test_time_in_total)
        summary = {
           'epoch': epoch+1, 
           'lr': lr_schedule(epoch+1), 
           'train time': train_time, 
           'train loss': sum_(train_stats['loss'])/N_train, 
           'train acc': sum_(train_stats['correct'])/N_train, 
           'test time': test_time, 
           'test loss': sum_(test_stats['loss'])/N_test, 
           'test acc': sum_(test_stats['correct'])/N_test,
           'total time': t.total_time, 
        }
        for logger in loggers:
            logger.append(summary)    
    return summary


#####################
## network visualisation (requires pydot)
#####################
class ColorMap(dict):
    palette = (
        'bebada,ffffb3,fb8072,8dd3c7,80b1d3,fdb462,b3de69,fccde5,bc80bd,ccebc5,ffed6f,1f78b4,33a02c,e31a1c,ff7f00,'
        '4dddf8,e66493,b07b87,4e90e3,dea05e,d0c281,f0e189,e9e8b1,e0eb71,bbd2a4,6ed641,57eb9c,3ca4d4,92d5e7,b15928'
    ).split(',')
    def __missing__(self, key):
        self[key] = self.palette[len(self) % len(self.palette)]
        return self[key]

def make_pydot(nodes, edges, direction='LR', sep=sep, **kwargs):
    import pydot
    parent = lambda path: path[:-1]
    stub = lambda path: path[-1]
    class Subgraphs(dict):
        def __missing__(self, path):
            subgraph = pydot.Cluster(sep.join(path), label=stub(path), style='rounded, filled', fillcolor='#77777744')
            self[parent(path)].add_subgraph(subgraph)
            return subgraph
    subgraphs = Subgraphs()
    subgraphs[()] = g = pydot.Dot(rankdir=direction, directed=True, **kwargs)
    g.set_node_defaults(
        shape='box', style='rounded, filled', fillcolor='#ffffff')
    for node, attr in nodes:
        path = tuple(node.split(sep))
        subgraphs[parent(path)].add_node(
            pydot.Node(name=node, label=stub(path), **attr))
    for src, dst, attr in edges:
        g.add_edge(pydot.Edge(src, dst, **attr))
    return g

get_params = lambda mod: {p.name: getattr(mod, p.name, '?') for p in signature(type(mod)).parameters.values()}


class DotGraph():
    colors = ColorMap()
    def __init__(self, net, size=15, direction='LR'):
        graph = build_graph(net)
        self.nodes = [(k, {
            'tooltip': '%s %.1000r' % (type(n).__name__, get_params(n)), 
            'fillcolor': '#'+self.colors[type(n)],
        }) for k, (n, i) in graph.items()] 
        self.edges = [(src, k, {}) for (k, (n, i)) in graph.items() for src in i]
        self.size, self.direction = size, direction

    def dot_graph(self, **kwargs):
        return make_pydot(self.nodes, self.edges, size=self.size, 
                            direction=self.direction, **kwargs)

    def svg(self, **kwargs):
        return self.dot_graph(**kwargs).create(format='svg').decode('utf-8')

    try:
        import pydot
        def _repr_svg_(self):
            return self.svg()
    except ImportError:
        def __repr__(self):
            return 'pydot is needed for network visualisation'

walk = lambda dict_, key: walk(dict_, dict_[key]) if key in dict_ else key
   
def remove_identity_nodes(net):  
    #remove identity nodes for more compact visualisations
    graph = build_graph(net)
    remap = {k: i[0] for k,(v,i) in graph.items() if isinstance(v, Identity)}
    return {k: (v, [walk(remap, x) for x in i]) for k, (v,i) in graph.items() if not isinstance(v, Identity)}

def cutout(mask_size, p, cutout_inside, mask_color=(0, 0, 0)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout