import os
import torch
import torch.nn as nn
from dataloader import KeyDataset
from torch.utils.data import DataLoader
from signal_process import signal_process
from time import time
from torch.utils.tensorboard import SummaryWriter
from torch import optim
# from torchsummary import summary
import numpy as np
import random


# TODO : IMPORTANT !!! Please change it to True when you submit your code
is_test_mode = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
n_epochs = 100
batch_size = 16
num_label = 24
method = 'logmelspectrogram'
sr = 22050

# continue training previously saved model
is_continue_mode = False

# reproduce result
seed = 929
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# Global Average Pooling
class GlobalAvgPooling(nn.Module):
    def __init__(self):
        super(GlobalAvgPooling, self).__init__()

    def forward(self, x):
        return x.mean(dim=-1, keepdim=False).mean(dim=-1, keepdim=False)


# TODO : Build your model here
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.75)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(512, 24, 1),
            nn.BatchNorm2d(24),
            nn.ReLU(True)
        )

        self.avgpool = GlobalAvgPooling()

        # He initialization for Convolutional layers & constant initialization for Batch normalization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        return x


# Save model with state_dict
def save_checkpoint(epoch, model, optimizer, path):
    state = {
        'epoch': epoch,
        'net_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict()
    }
    torch.save(state, path)


if not is_test_mode:

    # Write the result to tensorboard
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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # Learning rate scheduler: ReduceLROnPlateau
    scheduler_list = [
        optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-4),
        optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1)),
        optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.5),
        optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[2, 5, 10, 11, 28], gamma=0.5),
        optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96),
        optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=0)
    ]
    scheduler = scheduler_list[0]
    scheduler_name = scheduler.__class__.__name__

    if is_continue_mode:
        checkpoint = torch.load(restore_path)
        model.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])

    # to compare the result of current epoch with previous best accuracy
    top_accuracy = 0

    # Training and Validation
    for epoch in range(n_epochs):
        epoch_start = time()
        model.train()

        train_correct = 0
        train_loss = 0

        # print learning rate of current epoch
        lr = optimizer.param_groups[0]['lr']
        print('==== Epoch:', epoch, ', LR:', lr)

        for idx, (features, labels) in enumerate(train_loader):

            optimizer.zero_grad()
            features = signal_process(features, sr=sr, method=method).to(device)
            labels = labels.to(device)

            # # Summary model with summarywriter
            # input_shape = features.shape[1:]
            # summary(model, input_shape)

            output = model(features)
            loss = criterion(output, labels)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            preds = output.argmax(dim=-1, keepdim=True)
            train_correct += (preds.squeeze() == labels).float().sum()

        print("Train Loss: %.2f, Train Accuracy: %.3f" % (
            train_loss / len(train_loader), train_correct / len(train_dataset)))

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

        print("Valid Loss: %.2f, Valid Accuracy: %.3f" % (
            valid_loss / len(valid_loader), valid_correct / len(valid_dataset)))

        # Write to tensorboard
        writer.add_scalars('loss/training+validation', {"loss_training": train_loss / len(train_loader),
                                                        "loss_validation": valid_loss / len(valid_loader)}, epoch)
        writer.add_scalars('accuracy/training+validation', {"accuracy_training": train_correct / len(train_dataset),
                                                            "accuracy_validation": valid_correct / len(valid_dataset)}, epoch)
        writer.add_scalar('lr/{}'.format(scheduler_name), lr, epoch)

        # learning rate scheduler
        scheduler.step(metrics=valid_loss)

        train_accuracy = train_correct / len(train_dataset)
        valid_accuracy = valid_correct / len(valid_dataset)
        # Save the model of current epoch with train_accuracy and valid_accuracy
        torch.save(model, "./ckpt/%d_%.4f_%.4f.pth" % (epoch, train_accuracy, valid_accuracy))
        # Save the model which results the best accuracy
        if valid_accuracy > top_accuracy:
            torch.save(model, './ckpt/top_accuracy.pth')
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