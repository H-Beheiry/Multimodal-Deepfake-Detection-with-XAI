from torch import nn

# TODO: Dynamic architecture based on input size

class CNNnetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 5 conv blocks -> flatten layer -> linear layer 
        self.conv1= nn.Sequential(
            nn.Conv2d(in_channels=1,
                     out_channels=16,
                     kernel_size=3,
                     stride=1,
                     padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2= nn.Sequential(
            nn.Conv2d(in_channels=16,
                     out_channels=32,
                     kernel_size=3,
                     stride=1,
                     padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)            
        )
        self.conv3= nn.Sequential(
            nn.Conv2d(in_channels=32,
                     out_channels=64,
                     kernel_size=3,
                     stride=1,
                     padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4= nn.Sequential(
            nn.Conv2d(in_channels=64,
                     out_channels=128,
                     kernel_size=3,
                     stride=1,
                     padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5= nn.Sequential(
            nn.Conv2d(in_channels=128,
                     out_channels=256,
                     kernel_size=3,
                     stride=1,
                     padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten= nn.Flatten()
        self.linear= nn.Linear(33280,2)      
        
    def forward(self, input_data):
        x= self.conv1(input_data)
        x= self.conv2(x)
        x= self.conv3(x)
        x= self.conv4(x)
        x= self.conv5(x)
        x= self.flatten(x)
        logits= self.linear(x)
        return logits