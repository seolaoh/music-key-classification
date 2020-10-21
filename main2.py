import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import KeyDataset
from torch.utils.data import DataLoader
from signal_process import signal_process
import time
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# TODO : IMPORTANT !!! Please change it to True when you submit your code
is_test_mode = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO : IMPORTANT !!! Please specify the path where your best model is saved
# example : ckpt/model.pth
ckpt_dir = 'ckpt'
best_saved_model = 'model.pth'
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
restore_path = os.path.join(ckpt_dir, best_saved_model)

# Data paths
# TODO : IMPORTANT !!! Do not change metadata_path. Test will be performed by replacing this file.
metadata_path = 'metadata.csv'
audio_dir = 'audio'

# TODO : Declare additional hyperparameters
# not fixed (change or add hyperparameter as you like)
n_epochs = 100
batch_size = 16
num_label = 24
method = 'librosa_chroma_cqt'
sr = 22050



class GlobalAvgPooling(nn.Module):
    def __init__(self):
        super(GlobalAvgPooling, self).__init__()

    def forward(self, x):
        return x.mean(dim=-1, keepdim=False).mean(dim=-1, keepdim=False)


# TODO : Build your model here
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 20, 3, padding = 1),
            nn.BatchNorm2d(20),
            nn.ELU(True),
            nn.Conv2d(20, 20, 3,padding=  1),
            nn.BatchNorm2d(20),
            nn.ELU(True),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(20, 40, 3, padding = 1),
            nn.BatchNorm2d(40),
            nn.ELU(True),
            nn.Conv2d(40, 40, 3, padding = 1),
            nn.BatchNorm2d(40),
            nn.ELU(True),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(40, 80, 3, padding = 1),
            nn.BatchNorm2d(80),
            nn.ELU(True),
            nn.Conv2d(80, 80, 3, padding = 1),
            nn.BatchNorm2d(80),
            nn.ELU(True),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(80, 160, 3, padding = 1),
            nn.BatchNorm2d(160),
            nn.ELU(True),
            nn.Dropout(0.5),
            nn.Conv2d(160, 160, 3, padding = 1),
            nn.BatchNorm2d(160),
            nn.ELU(True),
            nn.Dropout(0.5),
            nn.Conv2d(160, 24, 3, padding = 1),
            nn.ELU(True),
            nn.BatchNorm2d(24)
        )

        self.avgpool = GlobalAvgPooling()
        # self.fc = nn.Linear(24, 24, bias=True)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        # x = self.fc(x)

        return x

if not is_test_mode:

    # Load Dataset and Dataloader
    train_dataset = KeyDataset(metadata_path=metadata_path, audio_dir=audio_dir, sr=sr, split='training')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = KeyDataset(metadata_path=metadata_path, audio_dir=audio_dir, sr=sr, split='validation')
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    # Define Model, loss, optimizer
    model = YourModel()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


    # Training and Validation
    for epoch in range(n_epochs):
        start = time.time()
        model.train()

        train_correct = 0
        train_loss = 0

        for idx, (features, labels) in enumerate(train_loader):

            optimizer.zero_grad()

            features = signal_process(features, sr=sr, method=method).to(device)
            labels = labels.to(device)

            output = model(features)

            loss = criterion(output, labels)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            preds = output.argmax(dim=-1, keepdim=True)
            train_correct += (preds.squeeze() == labels).float().sum()
            train_accuracy = train_correct / len(train_dataset)

        print("==== Epoch: %d, Train Loss: %.2f, Train Accuracy: %.3f" % (
            epoch, train_loss / len(train_loader), train_accuracy))

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
            valid_accuracy = valid_correct / len(valid_dataset)
        print("==== Epoch: %d, Valid Loss: %.2f, Valid Accuracy: %.3f" % (
            epoch, valid_loss / len(valid_loader), valid_accuracy))

        print("+++++ time:", time.time() - start)

        writer.add_scalar('train_accuaracy', train_accuracy, epoch)
        writer.add_scalar('valid_accuracy', valid_accuracy, epoch)
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