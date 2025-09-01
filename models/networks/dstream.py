import torch
import torch.nn.functional as F
import torch.nn as nn
from models.networks.base_network import BaseNetwork
from models.networks.DenseArchitecture import _DenseBlock, _Transition
from collections import OrderedDict
import torch.nn.utils.spectral_norm as spectral_norm

# Content/style stream.
# The two streams are symmetrical with the same network structure,
# aiming at extracting corresponding feature representations in different levels.
class Stream(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        growth_rate = 32
        bn_size = 4
        drop_rate = 0.0
        block_config = [2, 4, 8, 16, 16]
        num_init_features = 64
        self.features = nn.ModuleList()

         # First convolution
        self.block0 = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", spectral_norm(nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))),
                    # ("norm0", nn.InstanceNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )


         # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=False,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            # print(f"{num_features} + {num_layers} * {growth_rate} = {num_features + num_layers * growth_rate}")
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)    
                num_features = num_features // 2
        # Final instance norm
        # self.features.add_module("norm5", nn.InstanceNorm2d(num_features))

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def down(self, input):
        return F.interpolate(input, scale_factor=0.5)

    def forward(self,input):
        x0 = self.block0(input)

        x1 = self.features.denseblock1(x0)
        x1 = self.features.transition1(x1)

        x2 = self.features.denseblock2(x1)
        x2 = self.features.transition2(x2)

        x3 = self.features.denseblock3(x2)
        x3 = self.features.transition3(x3)

        x4 = self.features.denseblock4(x3)
        x4 = self.features.transition4(x4)

        x5 = self.features.denseblock5(x4)
        x5 = self.down(x5)


        return [x0, x1, x2, x3, x4, x5]

# Test above architecture with random input.
if __name__ == "__main__":
    import torch
    from options.train_options import TrainOptions

    # parse options
    opt = TrainOptions().parse()
    stream = Stream(opt=opt)
    input = torch.randn(1, 3, 512, 512)
    x0, x1, x2, x3, x4, x5 = stream(input)
    print(x0.shape, x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
