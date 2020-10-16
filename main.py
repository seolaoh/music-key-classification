import os
import torch
import torch.nn as nn
from dataloader import KeyDataset
from torch.utils.data import DataLoader
from signal_process import signal_process
from time import time
# from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter



# TODO : IMPORTANT !!! Please change it to True when you submit your code
is_test_mode = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# TODO : IMPORTANT !!! Please specify the path where your best model is saved
# example : ckpt/model.pth
ckpt_dir = 'ckpt'
best_saved_model = 'last.pth'
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


# TODO : Build your model here
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 20, 5),
            nn.BatchNorm2d(20),
            nn.ELU(True),
            nn.Conv2d(20, 20, 3),
            nn.BatchNorm2d(20),
            nn.ELU(True),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(20, 40, 3),
            nn.BatchNorm2d(40),
            nn.ELU(True),
            nn.Conv2d(40, 40, 3),
            nn.BatchNorm2d(40),
            nn.ELU(True),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(40, 80, 3),
            nn.BatchNorm2d(80),
            nn.ELU(True),
            nn.Conv2d(80, 80, 3),
            nn.BatchNorm2d(80),
            nn.ELU(True),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(80, 160, 3),
            nn.BatchNorm2d(160),
            nn.ELU(True),
            nn.Dropout(0.5),
            nn.Conv2d(160, 160, 3),
            nn.BatchNorm2d(160),
            nn.ELU(True),
            nn.Dropout(0.5),
            nn.Conv2d(160, 24, 1),
            nn.BatchNorm2d(24),
            nn.ELU(True)
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
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    # Define Model, loss, optimizer
    model = YourModel()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # # Summary model by summarywriter
    # input_shape = (batch_size, )

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