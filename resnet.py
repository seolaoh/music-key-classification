import os
import torch
import torch.nn as nn
from dataloader import KeyDataset
from torch.utils.data import DataLoader
from signal_process import signal_process
from time import time
from torch.utils.tensorboard import SummaryWriter




# TODO : IMPORTANT !!! Please change it to True when you submit your code
is_test_mode = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# TODO : IMPORTANT !!! Please specify the path where your best model is saved
# example : ckpt/model.pth
ckpt_dir = 'ckpt'
best_saved_model = 'top_accuracy.pth'
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
restore_path = os.path.join(ckpt_dir, best_saved_model)

# Data paths
# TODO : IMPORTANT !!! Do not change metadata_path. Test will be performed by replacing this file.
metadata_path = 'metadata.csv'
audio_dir = 'audio'

# TODO : Declare additional hyperparameters
# not fixed (change or add hyperparameter as you like)
is_continue_mode = False
n_epochs = 100
batch_size = 16
num_label = 24
method = 'logmelspectrogram'
sr = 22050


class GlobalAvgPooling(nn.Module):
    def __init__(self):
        super(GlobalAvgPooling, self).__init__()

    def forward(self, x):
        return x.mean(dim=-1, keepdim=False).mean(dim=-1, keepdim=False)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        identity = x

        out = self.conv1(x) # 3x3 stride = stride
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # 3x3 stride = 1
        out = self.bn2(out)

        if self.downsample is not None: # stride=2 일 경우 identity와 out의 shape이 다른 문제 해결하기 위함
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes) #conv1x1(64,64)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)#conv3x3(64,64)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion) #conv1x1(64,256) channel 4배 되기 때문에
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x) # 1x1 stride = 1
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # 3x3 stride = stride
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out) # 1x1 stride = 1
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class YourModel(nn.Module):
    # model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs) #resnet 50
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(YourModel, self).__init__()

        self.inplanes = 64

        # inputs = 3*224*224
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # outputs = self.conv1(inputs)
        # outputs.shape = [64*112*112]

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # outputs = [64*56*56]

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:  # zero_init_residual이 True이면
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    # self.inplanes = 256
    # self.layer1 = self._make_layer(Bottleneck, 64, layers[0]'''3''')
    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None

        # downsample을 여기서는 channel size 맞추기 위해서도 사용
        if stride != 1 or self.inplanes != planes * block.expansion:  # 64 != 256

            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),  # conv1x1(64, 256, stride=1)
                nn.BatchNorm2d(planes * block.expansion),  # batchnrom2d(256)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        # layers.append(Bottleneck(64, 64, 1, downsample))

        self.inplanes = planes * block.expansion  # self.inplanes = 256

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))  # * 2

        return nn.Sequential(*layers)  # layers에 쌓여 있는 것 return

    # self.layer1 = [
    # Bottleneck(64, 64, 1, downsample)
    # Bottleneck(256, 64)
    # Bottleneck(256, 64)
    # ]

    def forward(self, x):
        x = self.conv1(x.unsqueeze(1))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # global average pooling: fully connected 어느 정도 대체. 1*1으로 평균 내서 묶어버림
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def save_checkpoint(epoch, model, optimizer, path):
    state = {
        'epoch': epoch,
        'net_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict()
    }
    torch.save(state, path)


if not is_test_mode:

    writer = SummaryWriter('./tensorboard')

    # Load Dataset and Dataloader
    train_dataset = KeyDataset(metadata_path=metadata_path, audio_dir=audio_dir, sr=sr, split='training')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = KeyDataset(metadata_path=metadata_path, audio_dir=audio_dir, sr=sr, split='validation')
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Define Model, loss, optimizer
    model = YourModel(Bottleneck, [3, 4, 6, 3])
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.95)


    if is_continue_mode:
        checkpoint = torch.load(restore_path)
        model.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])

    top_accuracy = 0

    # Training and Validation
    for epoch in range(n_epochs):
        epoch_start = time()
        model.train()

        train_correct = 0
        train_loss = 0

        for idx, (features, labels) in enumerate(train_loader):

            optimizer.zero_grad()
            # print(features.shape)
            features = signal_process(features, sr=sr, method=method).to(device)
            # print(features.shape)
            labels = labels.to(device)

            output = model(features)
            loss = criterion(output, labels)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            preds = output.argmax(dim=-1, keepdim=True)
            train_correct += (preds.squeeze() == labels).float().sum()

        print("==== Epoch: %d, Train Loss: %.2f, Train Accuracy: %.3f" % (
            epoch, train_loss / len(train_loader), train_correct / len(train_dataset)))

        model.eval()

        valid_correct = 0
        valid_loss = 0

        for idx, (features, labels) in enumerate(valid_loader):
            features = signal_process(features, sr=sr, method=method).to(device)
            labels = labels.to(device)

            output = model(features)
            loss = criterion(output, labels)
            valid_loss += loss.item()

            preds = output.argmax(dim=-1, keepdim=True)
            valid_correct += (preds.squeeze() == labels).float().sum()

        print("==== Epoch: %d, Valid Loss: %.2f, Valid Accuracy: %.3f" % (
            epoch, valid_loss / len(valid_loader), valid_correct / len(valid_dataset)))

        # Write to tensorboard
        writer.add_scalars('loss/training+validation', {"loss_training": train_loss / len(train_loader),
                                                        "loss_validation": valid_loss / len(valid_loader)}, epoch)
        writer.add_scalars('accuracy/training+validation', {"accuracy_training": train_correct / len(train_dataset),
                                                            "accuracy_validation": valid_correct / len(valid_dataset)}, epoch)

        save_checkpoint(epoch, model, optimizer, "./ckpt/last.pth")
        # torch.save(model, "./ckpt/last.pth")
        valid_accuracy = valid_correct / len(valid_dataset)
        if valid_accuracy > top_accuracy:
            save_checkpoint(epoch, model, optimizer, "./ckpt/top_accuracy.pth")
            # torch.save(model, './ckpt/top_accuracy.pth')
            top_accuracy = valid_accuracy
        print('TIME: %6.3f' % (time() - epoch_start))

    writer.close()

elif is_test_mode:

    # Load Dataset and Dataloader
    test_dataset = KeyDataset(metadata_path=metadata_path, audio_dir=audio_dir, sr=sr, split='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Restore model
    model = torch.load(restore_path).to(device)
    print('==== Model restored : %s' % restore_path)

    # TODO: IMPORTANT!! MUST CALCULATE ACCURACY ! You may change this part, but must print accuracy in the right manner

    test_correct = 0

    for features, labels in test_loader:
        features = signal_process(features, sr=sr, method=method).to(device)
        labels = labels.to(device)

        output = model(features)

        preds = output.argmax(dim=-1, keepdim=True)
        test_correct += (preds.squeeze() == labels).float().sum()

    print("=== Test accuracy: %.3f" % (test_correct / len(test_dataset)))