# model.py

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, bias=False)
        self.fc1 = nn.Linear(4096, 50, bias=False) # corr 320
        self.fc2 = nn.Linear(50, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # rem2
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x)) # rem2
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(x.size(0), -1) #, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# model for iteration05 in session07 assignment
class Net_Iter05(nn.Module):
    def __init__(self):
        super(Net_Iter05, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.15),
            nn.Conv2d(8, 8, 3, stride=1, padding=1),
            #
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.15),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.15),
            nn.MaxPool2d(2, 2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 12, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(0.15),
            nn.Conv2d(12, 12, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(0.15),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(12, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.15),
            nn.MaxPool2d(2, 2),
        )


        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gap(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)

        x = F.log_softmax(x, dim=1)
        return x


# model for BatchNormalization session08 assignment
class Net_BatchNorm(nn.Module):
    def __init__(self):
        super(Net_BatchNorm, self).__init__()

        # C1 C2
        self.conv_C1_C2 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.15),
            # C1
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.15),
            # C2
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.15),
        )

        # c3 P1
        self.conv_c3_P1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.MaxPool2d(2, 2),
        )

        # C4 C5 C6
        self.conv_C4_C5_C6 = nn.Sequential(
            # C4
            nn.Conv2d(10, 24, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(0.15),
            # C5
            nn.Conv2d(24, 24, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(0.15),
            # C6
            nn.Conv2d(24, 24, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(0.15),
        )

        # c7 P2
        self.conv_c7_P2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.MaxPool2d(2, 2),
        )

        # C8 C9 C10
        self.conv_C8_C9_C10 = nn.Sequential(
            # C8
            nn.Conv2d(10, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.15),
            # C9
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.15),
            # C10
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.15),
        )

        # GAP  # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        )
        # C11
        self.conv_C11 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )


    def forward(self, x):
        x = self.conv_C1_C2(x)
        x = self.conv_c3_P1(x)
        x = self.conv_C4_C5_C6(x)
        x = self.conv_c7_P2(x)
        x = self.conv_C8_C9_C10(x)
        x = self.gap(x)
        x = self.conv_C11(x)
        x = x.view(x.size(0), -1)

        x = F.log_softmax(x, dim=1)
        return x


# group normalization for session08 assignment
class Net_GroupNorm(nn.Module):
    def __init__(self):
        super(Net_GroupNorm, self).__init__()

        # C1 C2
        self.conv_C1_C2 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(4, 16),
            nn.Dropout(0.1),
            # C1
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(4, 16),
            nn.Dropout(0.1),
            # C2
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(4, 16),
            nn.Dropout(0.1),
        )

        # c3 P1
        self.conv_c3_P1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.MaxPool2d(2, 2),
        )

        # C4 C5 C6
        self.conv_C4_C5_C6 = nn.Sequential(
            # C4
            nn.Conv2d(10, 24, 3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(4, 24),
            nn.Dropout(0.1),
            # C5
            nn.Conv2d(24, 24, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(4, 24),
            nn.Dropout(0.1),
            # C6
            nn.Conv2d(24, 24, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(4, 24),
            nn.Dropout(0.1),
        )

        # c7 P2
        self.conv_c7_P2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.MaxPool2d(2, 2),
        )

        # C8 C9 C10
        self.conv_C8_C9_C10 = nn.Sequential(
            # C8
            nn.Conv2d(10, 32, 3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(4, 32),
            nn.Dropout(0.1),
            # C9
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(4, 32),
            nn.Dropout(0.1),
            # C10
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(4, 32),
            nn.Dropout(0.1),
        )

        # GAP  # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        )
        # C11
        self.conv_C11 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )


    def forward(self, x):
        x = self.conv_C1_C2(x)
        x = self.conv_c3_P1(x)
        x = self.conv_C4_C5_C6(x)
        x = self.conv_c7_P2(x)
        x = self.conv_C8_C9_C10(x)
        x = self.gap(x)
        x = self.conv_C11(x)
        x = x.view(x.size(0), -1)

        x = F.log_softmax(x, dim=1)
        return x

# layer normalization for session08 assignment

class Net_LayerNorm(nn.Module):
    def __init__(self, input_size):
        super(Net_LayerNorm, self).__init__()

        self.input_size = input_size

        self.conv_C1_C2 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            #nn.LayerNorm([16, 32, 32]), # <--- Normalize activations over C, H, and W
            nn.GroupNorm(num_groups=1, num_channels=16),  # Equivalent to LayerNorm
            nn.Dropout(0.15),
            # C1
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            #nn.LayerNorm([16, 32, 32]), # <--- Normalize activations over C, H, and W
            nn.GroupNorm(num_groups=1, num_channels=16),  # Equivalent to LayerNorm
            nn.Dropout(0.15),
            # C2
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            #nn.LayerNorm([16, 32, 32]), # <--- Normalize activations over C, H, and W
            nn.GroupNorm(num_groups=1, num_channels=16),  # Equivalent to LayerNorm
            nn.Dropout(0.15),
        )

        # c3 P1
        self.conv_c3_P1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.MaxPool2d(2, 2),
        )

        # C4 C5 C6
        self.conv_C4_C5_C6 = nn.Sequential(
            # C4
            nn.Conv2d(10, 28, 3, padding=1),
            nn.ReLU(),
            #nn.LayerNorm([28, 16, 16]), # <--- Normalize activations over C, H, and W
            nn.GroupNorm(num_groups=1, num_channels=28),  # Equivalent to LayerNorm
            nn.Dropout(0.15),
            # C5
            nn.Conv2d(28, 28, 3, stride=1, padding=1),
            nn.ReLU(),
            #nn.LayerNorm([28, 16, 16]), # <--- Normalize activations over C, H, and W
            nn.GroupNorm(num_groups=1, num_channels=28),  # Equivalent to LayerNorm
            nn.Dropout(0.15),
            # C6
            nn.Conv2d(28, 28, 3, stride=1, padding=1),
            nn.ReLU(),
            #nn.LayerNorm([28, 16, 16]), # <--- Normalize activations over C, H, and W
            nn.GroupNorm(num_groups=1, num_channels=28),  # Equivalent to LayerNorm
            nn.Dropout(0.15),
        )

        # c7 P2
        self.conv_c7_P2 = nn.Sequential(
            nn.Conv2d(in_channels=28, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.MaxPool2d(2, 2),
        )

        # C8 C9 C10
        self.conv_C8_C9_C10 = nn.Sequential(
            # C8
            nn.Conv2d(10, 36, 3, padding=1),
            nn.ReLU(),
            #nn.LayerNorm([36, 8, 8]), # <--- Normalize activations over C, H, and W
            nn.GroupNorm(num_groups=1, num_channels=36),  # Equivalent to LayerNorm
            nn.Dropout(0.15),
            # C9
            nn.Conv2d(36, 36, 3, stride=1, padding=1),
            nn.ReLU(),
            #nn.LayerNorm([36, 8, 8]), # <--- Normalize activations over C, H, and W
            nn.GroupNorm(num_groups=1, num_channels=36),  # Equivalent to LayerNorm
            nn.Dropout(0.15),
            # C10
            nn.Conv2d(36, 36, 3, stride=1, padding=1),
            nn.ReLU(),
            #nn.LayerNorm([36, 8, 8]), # <--- Normalize activations over C, H, and W
            nn.GroupNorm(num_groups=1, num_channels=36),  # Equivalent to LayerNorm
            nn.Dropout(0.15),
        )

        # GAP  # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        )
        # C11
        self.conv_C11 = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )


    def forward(self, x):
        x = self.conv_C1_C2(x)
        x = self.conv_c3_P1(x)
        x = self.conv_C4_C5_C6(x)
        x = self.conv_c7_P2(x)
        x = self.conv_C8_C9_C10(x)
        x = self.gap(x)
        x = self.conv_C11(x)
        x = x.view(x.size(0), -1)

        x = F.log_softmax(x, dim=1)
        return x


