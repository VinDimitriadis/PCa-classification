def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures that CUDA convolution is deterministic

seed = 42 #
seed_everything(seed)
#device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")


def resizeTensor(x, scale_factor=None, size=None):
   
    if len(x.shape) == 3:
        return F.interpolate(x, scale_factor=scale_factor, size=size,
                             mode='linear',
                             align_corners=True)
    if len(x.shape) == 4:
        return F.interpolate(x, scale_factor=scale_factor, size=size,
                             mode='bicubic',
                             align_corners=True)
    elif len(x.shape) == 5:
        return F.interpolate(x, scale_factor=scale_factor, size=size,
                             mode='trilinear',
                             align_corners=True)
def tensor2array(tensor):
    return tensor.data.cpu().numpy()
    
class GlobalAvgPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        
        if len(inputs.shape) == 2:
            return inputs
        elif len(inputs.shape) == 3:
            return nn.functional.adaptive_avg_pool1d(inputs, 1).view(inputs.size(0), -1)
        elif len(inputs.shape) == 4:
            return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)
        elif len(inputs.shape) == 5:
            return nn.functional.adaptive_avg_pool3d(inputs, 1).view(inputs.size(0), -1)
        
class GlobalMaxPool(nn.Module):
    def __init__(self):
        
        super().__init__()
    def forward(self, inputs):
        
        if len(inputs.shape) == 2:
            return inputs
        elif len(inputs.shape) == 3:
            return nn.functional.adaptive_max_pool1d(inputs, 1).view(inputs.size(0), -1)
        elif len(inputs.shape) == 4:
            return nn.functional.adaptive_max_pool2d(inputs, 1).view(inputs.size(0), -1)
        elif len(inputs.shape) == 5:
            return nn.functional.adaptive_max_pool3d(inputs, 1).view(inputs.size(0), -1)
        
class GlobalMaxAvgPool(nn.Module):
    def __init__(self):
        
        super().__init__()
        self.GAP = GlobalAvgPool()
        self.GMP = GlobalMaxPool()
    def forward(self, inputs):
        return (self.GMP(inputs) + self.GAP(inputs))/2.
    


def MakeNorm(dim, channel, norm='bn', gn_c=8):
    
    if norm == 'bn':
        if dim == 1:
            return nn.BatchNorm1d(channel)
        elif dim == 2:
            return nn.BatchNorm2d(channel)
        elif dim == 3:
            return nn.BatchNorm3d(channel)
    elif norm == 'in':
        if dim == 1:
            return nn.InstanceNorm1d(channel)
        elif dim == 2:
            return nn.InstanceNorm2d(channel)
        elif dim == 3:
            return nn.InstanceNorm3d(channel)
    elif norm == 'gn':
        return nn.GroupNorm(gn_c, channel)
    elif norm == 'None' or norm is None:
        return nn.Identity()
def MakeActive(active='relu'): #selu/gelu
    
    if active == 'relu':
        return nn.ReLU(inplace=True)
    elif active == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif active == 'selu': #selu looks better
        return nn.SELU(inplace=True)
    elif active == 'gelu':
        return nn.GELU()
    elif active == 'None' or active is None:
        return nn.Identity()
    else:
        raise ValueError('should be relu or leakyrelu')


def MakeConv(in_channels, out_channels, kernel_size, padding=1, stride=1, dim=2, bias=False):
    if dim == 1:
        return nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias)
    elif dim == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias)
    elif dim == 3:
        return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=bias)
    
class ConvNormActive(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size = 3, norm='bn', active='relu', gn_c = 8, dim = 3, padding = 1, dropout=0.5):

        super().__init__()

        self.conv = MakeConv(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dim = dim)
        self.norm = MakeNorm(dim, out_channels, norm, gn_c)
        self.active = MakeActive(active)
        self.dropout = nn.Dropout(dropout) #dropout method 2 after activations
        
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.active(out)
        out = self.dropout(out) #dropout method 2 after activation 
        return out

class VGGBlock(ConvNormActive):
    
    def __init__(self, *args, **kwarg):
        super().__init__(*args, **kwarg)

class VGGStage(nn.Module):
    def __init__(self, in_channels, out_channels, block_num=2, norm='bn', active='relu', gn_c=8, dim=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_num = block_num
        self.dim = dim
        self.block_list = nn.ModuleList([])
        for index in range(self.block_num):
            if index == 0:
                self.block_list.append(VGGBlock(in_channels, out_channels, norm=norm, active=active, gn_c = gn_c, dim = dim))
            else:
                self.block_list.append(VGGBlock(out_channels, out_channels, norm=norm, active=active, gn_c = gn_c, dim = dim))

    def forward(self, x):
      
        for block in self.block_list:
            x = block(x)
        return x

class ClassificationHead(nn.Module):
    
    def __init__(self, label_category_dict, in_channel, bias=True):
        super().__init__()
        self.classification_head = torch.nn.ModuleDict({})
        for key in label_category_dict.keys():
            self.classification_head[key] = torch.nn.Linear(in_channel, label_category_dict[key], bias=bias)

    def forward(self, f):
        
        logits = {}
        if isinstance(f,dict):
            print('dict element-wised forward')
            for key in self.classification_head.keys():
                logits[key] = self.classification_head[key](f[key])
        elif isinstance(f,list):
            print('list element-wised forward')
            for key_index, key in enumerate(self.classification_head.keys()):
                logits[key] = self.classification_head[key](f[key_index])
        else:
            for key in self.classification_head.keys():
                logits[key] = self.classification_head[key](f)
        return logits


class VGGEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 stage_output_channels=[64, 128, 256, 512],
                 blocks=[6, 12, 24, 16],
                 downsample_ration=[0.5, 0.5, 0.5, 0.5],
                 downsample_first=False,
                 norm='bn',
                 active='relu',# selu
                 gn_c=8,
                 dim=3):
        super().__init__()

        self.dim = dim
        self.blocks = blocks
        self.downsample_ration =downsample_ration
        self.downsample_first =downsample_first

        # res stages
        self.vgg_stage = nn.ModuleList([])
        for stage_index in range(len(stage_output_channels)):
            if stage_index == 0:
                self.vgg_stage.append(VGGStage(in_channels, stage_output_channels[stage_index], block_num=blocks[stage_index], norm=norm, active=active, gn_c=gn_c, dim=dim))
            else:
                self.vgg_stage.append(VGGStage(stage_output_channels[stage_index-1], stage_output_channels[stage_index], block_num=blocks[stage_index], norm=norm, active=active, gn_c=gn_c, dim=dim))

    def forward(self, x):
        f = x
        stage_features = []
        for stage_index, downsample_ratio in enumerate(self.downsample_ration):
            if self.downsample_first:
                f = resizeTensor(f, scale_factor=downsample_ratio)
                f = self.vgg_stage[stage_index](f)
            else:
                f = self.vgg_stage[stage_index](f)
                f = resizeTensor(f, scale_factor=downsample_ratio)
            stage_features.append(f)

        return stage_features