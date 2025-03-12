import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, skip_kernel_size, stride):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=skip_kernel_size, stride=stride, padding=skip_kernel_size//2, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x += self.shortcut(identity)
        x = F.relu(x)
        return x


class ResNetBasicBlock(nn.Module):
    def __init__(self, blocks_per_layer, channels_per_layer, kernels_per_layer, skip_kernels_per_layer, starting_input_channels, name, num_classes=10):
        super(ResNetBasicBlock, self).__init__()
        self.in_channels = channels_per_layer[0]
        self.name = name
        self.pool_size = 1


        self.input_layer = nn.Sequential(
            nn.Conv2d(starting_input_channels, self.in_channels, kernel_size=kernels_per_layer[0], stride=1, padding=kernels_per_layer[0]//2, bias=False),
            nn.BatchNorm2d(self.in_channels)
        )

        self.residual_layers = self._make_layers(blocks_per_layer, channels_per_layer, kernels_per_layer, skip_kernels_per_layer)


        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pool_size),
            nn.Flatten(),
            nn.Linear(channels_per_layer[-1], num_classes)
        )


    def _make_layer(self, out_channels, num_blocks, kernel_size, skip_kernel_size, stride):
        blocks = []
        #strides = []
        for i in range(num_blocks):
            if i != 0:
                stride = 1

            blocks.append(BasicBlock(self.in_channels, out_channels, kernel_size, skip_kernel_size, stride))
            #strides.append(stride)
            self.in_channels = out_channels
        
        #print(strides, '\n')

        return nn.Sequential(*blocks)



    def _make_layers(self, blocks_per_layer, channels_per_layer, kernels_per_layer, skip_kernels_per_layer):
        layers = []
        for i in range(len(blocks_per_layer)):
            if i == 0:
                stride = 1
            else: 
                stride = 2

            layers.append(
                self._make_layer(
                    out_channels = channels_per_layer[i],
                    num_blocks = blocks_per_layer[i],
                    kernel_size = kernels_per_layer[i],
                    skip_kernel_size = skip_kernels_per_layer[i],
                    stride = stride
                )
            )

        return nn.Sequential(*layers)



    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.residual_layers(x)
        x = self.output_layer(x)
        return x



def create_basicblock_model(blocks_per_layer, channels_per_layer, kernels_per_layer, skip_kernels_per_layer, starting_input_channels, name):
    return ResNetBasicBlock(blocks_per_layer, channels_per_layer, kernels_per_layer, skip_kernels_per_layer, starting_input_channels, name)


if __name__ == "__main__":
    model = create_basicblock_model(
        name = 'ResNet_v2',
        starting_input_channels = 3,
        blocks_per_layer = [8 , 13, 17, 12],
        channels_per_layer = [16, 32, 64, 128],
        kernels_per_layer = [3, 3, 3, 3],
        skip_kernels_per_layer = [1, 1, 1, 1]
        
    )
    
    print('Total model parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    summary(model, (3, 32, 32))

