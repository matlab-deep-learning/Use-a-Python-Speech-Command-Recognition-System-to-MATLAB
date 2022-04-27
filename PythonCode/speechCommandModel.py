import torch.nn as nn
import torch

NumF = 12
numHops = 98
timePoolSize = 13
dropoutProb = 0.2
numClasses = 11

class CNN(nn.Module):

    # Contructor
    def __init__(self, out_1=NumF):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(out_1)
        self.relu1 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=2 * out_1, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(2 * out_1)
        self.relu2 = nn.ReLU()

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.cnn3 = nn.Conv2d(in_channels=2 * out_1, out_channels=4 * out_1, kernel_size=3, padding=1)
        self.batch3 = nn.BatchNorm2d(4 * out_1)
        self.relu3 = nn.ReLU()

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.cnn4 = nn.Conv2d(in_channels=4 * out_1, out_channels=4 * out_1, kernel_size=3, padding=1)
        self.batch4 = nn.BatchNorm2d(4 * out_1)
        self.relu4 = nn.ReLU()
        self.cnn5 = nn.Conv2d(in_channels=4 * out_1, out_channels=4 * out_1, kernel_size=3, padding=1)
        self.batch5 = nn.BatchNorm2d(4 * out_1)
        self.relu5 = nn.ReLU()

        self.maxpool4 = nn.MaxPool2d(kernel_size=(timePoolSize, 1))

        self.dropout = nn.Dropout2d(dropoutProb)

        self.fc = nn.Linear(336, numClasses)

    # Prediction
    def forward(self, x):

        out = self.cnn1(x)
        out = self.batch1(out)
        out = self.relu1(out)

        out = self.maxpool1(out)

        out = self.cnn2(out)
        out = self.batch2(out)
        out = self.relu2(out)

        out = self.maxpool2(out)

        out = self.cnn3(out)
        out = self.batch3(out)
        out = self.relu3(out)

        out = self.maxpool3(out)

        out = self.cnn4(out)
        out = self.batch4(out)
        out = self.relu4(out)
        out = self.cnn5(out)
        out = self.batch5(out)
        out = self.relu5(out)

        out = self.maxpool4(out)

        out = self.dropout(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out