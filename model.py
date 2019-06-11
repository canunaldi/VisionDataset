# Feel free to change / extend / adapt this source code as needed to complete the homework, based on its requirements.
# This code is given as a starting point.
#
# REFEFERENCES
# The code is partly adapted from pytorch tutorials, including https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# ---- hyper-parameters ----
# You should tune these hyper-parameters using:
# (i) your reasoning and observations,
# (ii) by tuning it on the validation set, using the techniques discussed in class.
# You definitely can add more hyper-parameters here.
batch_size = 8
max_num_epoch = 100
hps = {'lr': 0.001}
decider = 0
# ---- options ----
DEVICE_ID = 'cuda'  # set to 'cpu' for cpu, 'cuda' / 'cuda:0' or similar for gpu.
LOG_DIR = 'checkpoints'
VISUALIZE = False  # set True to visualize input, prediction and the output from the last batch
LOAD_CHKPT = False

# --- imports ---
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import hw3utils

torch.multiprocessing.set_start_method('spawn', force=True)


# ---- utility functions -----
def get_loaders(batch_size, device):
    data_root = 'ceng483-s19-hw3-dataset'
    train_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root, 'train'), device=device)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root, 'val'), device=device)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    # Note: you may later add test_loader to here.
    return train_loader, val_loader


# ---- ConvNet -----
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.Conv2 = nn.Conv2d(16, 3, 5, padding=2)
        '''
        self.bb8 = torch.nn.BatchNorm2d(8)
        self.bb4 = torch.nn.BatchNorm2d(4)
        self.bb2 = torch.nn.BatchNorm2d(2)
        #1 layer
        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)
        self.conv1_2 = nn.Conv2d(1, 3, 5, padding=2)
        #2 layers
        self.conv2layer1 = nn.Conv2d(1,2,5,padding=2)
        self.conv2layer2 = nn.Conv2d(2,3,5,padding=2)
        #2 layers v2
        self.conv2_2layer1 = nn.Conv2d(1,4,5,padding=2)
        self.conv2_2layer2 = nn.Conv2d(4,3,5,padding=2)
        #2 layers size 3
        
        # 2 layers size 3
        self.conv2_2size3layer1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2_2size3layer2 = nn.Conv2d(8, 3, 3, padding=1)
        
        # 4 layers
        self.conv4layer1 = nn.Conv2d(1,4,5,padding=2)
        self.conv4layer2 = nn.Conv2d(4,16,5,padding=2)
        self.conv4layer3 = nn.Conv2d(16,4,5,padding=2)
        self.conv4layer4 = nn.Conv2d(4,3,5,padding=2)
        # 4 v2
        self.conv4_2layer1 = nn.Conv2d(1,8,5,padding=2)
        self.conv4_2layer2 = nn.Conv2d(8,32,5,padding=2)
        self.conv4_2layer3 = nn.Conv2d(32,8, 5,padding=2)
        self.conv4_2layer4 = nn.Conv2d(8,3,5,padding=2)

        # 4 layers size3
        self.conv4layer1size3 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv4layer2size3 = nn.Conv2d(4, 16, 3, padding=1)
        self.conv4layer3size3 = nn.Conv2d(16, 4, 3, padding=1)
        self.conv4layer4size3 = nn.Conv2d(4, 3, 3, padding=1)
        # 4 v2 size3
        self.conv4_2layer1size3 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv4_2layer2size3 = nn.Conv2d(8, 32, 3, padding=1)
        self.conv4_2layer3size3 = nn.Conv2d(32, 8, 3, padding=1)
        self.conv4_2layer4size3 = nn.Conv2d(8, 3, 3, padding=1)
        
        # 4 layers
        # no1
        self.conv4layer1no1 = nn.Conv2d(1, 2, 3, padding=1)
        self.conv4layer2no1 = nn.Conv2d(2, 2, 3, padding=1)
        self.conv4layer3no1 = nn.Conv2d(2, 8, 3, padding=1)
        self.conv4layer4no1 = nn.Conv2d(8, 3, 3, padding=1)

        # no2
        self.conv4layer1no2 = nn.Conv2d(1, 2, 5, padding=2)
        self.conv4layer2no2 = nn.Conv2d(2, 4, 5, padding=2)
        self.conv4layer3no2 = nn.Conv2d(4, 8, 5, padding=2)
        self.conv4layer4no2 = nn.Conv2d(8, 3, 5, padding=2)

        # no3
        self.conv4layer1no3 = nn.Conv2d(1, 2, 5, padding=2)
        self.conv4layer2no3 = nn.Conv2d(2, 8, 5, padding=2)
        self.conv4layer3no3 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv4layer4no3 = nn.Conv2d(8, 3, 5, padding=2)

        # no4
        self.conv4layer1no4 = nn.Conv2d(1, 4, 5, padding=2)
        self.conv4layer2no4 = nn.Conv2d(4, 8, 5, padding=2)
        self.conv4layer3no4 = nn.Conv2d(8, 4, 5, padding=2)
        self.conv4layer4no4 = nn.Conv2d(4, 3, 5, padding=2)

        # no5
        self.conv4layer1no5 = nn.Conv2d(1, 4, 5, padding=2)
        self.conv4layer2no5 = nn.Conv2d(4, 8, 5, padding=2)
        self.conv4layer3no5 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv4layer4no5 = nn.Conv2d(8, 3, 5, padding=2)

        # no6
        self.conv4layer1no6 = nn.Conv2d(1, 8, 5, padding=2)
        self.conv4layer2no6 = nn.Conv2d(8, 2, 5, padding=2)
        self.conv4layer3no6 = nn.Conv2d(2, 8, 5, padding=2)
        self.conv4layer4no6 = nn.Conv2d(8, 3, 5, padding=2)

        # no7
        self.conv4layer1no7 = nn.Conv2d(1, 8, 5, padding=2)
        self.conv4layer2no7 = nn.Conv2d(8, 4, 5, padding=2)
        self.conv4layer3no7 = nn.Conv2d(4, 8, 5, padding=2)
        self.conv4layer4no7 = nn.Conv2d(8, 3, 5, padding=2)

        # no8
        self.conv4layer1no8 = nn.Conv2d(1, 8, 5, padding=2)
        self.conv4layer2no8 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv4layer3no8 = nn.Conv2d(8, 8, 5, padding=2)
        self.conv4layer4no8 = nn.Conv2d(8, 3, 5, padding=2)

        # no9
        self.conv4layer1no9 = nn.Conv2d(1, 2, 3, padding=1)
        self.conv4layer2no9 = nn.Conv2d(2, 4, 3, padding=1)
        self.conv4layer3no9 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv4layer4no9 = nn.Conv2d(8, 3, 3, padding=1)

        # no10
        self.conv4layer1no10 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv4layer2no10 = nn.Conv2d(4, 2, 3, padding=1)
        self.conv4layer3no10 = nn.Conv2d(2, 8, 3, padding=1)
        self.conv4layer4no10 = nn.Conv2d(8, 3, 3, padding=1)

        # no11
        self.conv4layer1no11 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv4layer2no11 = nn.Conv2d(4, 4, 3, padding=1)
        self.conv4layer3no11 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv4layer4no11 = nn.Conv2d(8, 3, 3, padding=1)

        # no12
        self.conv4layer1no12 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv4layer2no12 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv4layer3no12 = nn.Conv2d(8, 4, 3, padding=1)
        self.conv4layer4no12 = nn.Conv2d(4, 3, 3, padding=1)

        # no13
        self.conv4layer1no13 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv4layer2no13 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv4layer3no13 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv4layer4no13 = nn.Conv2d(8, 3, 3, padding=1)

        # no14
        self.conv4layer1no14 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv4layer2no14 = nn.Conv2d(8, 2, 3, padding=1)
        self.conv4layer3no14 = nn.Conv2d(2, 8, 3, padding=1)
        self.conv4layer4no14 = nn.Conv2d(8, 3, 3, padding=1)

        # no15
        self.conv4layer1no15 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv4layer2no15 = nn.Conv2d(8, 4, 3, padding=1)
        self.conv4layer3no15 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv4layer4no15 = nn.Conv2d(8, 3, 3, padding=1)

        # no16
        self.conv4layer1no16 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv4layer2no16 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv4layer3no16 = nn.Conv2d(8, 4, 3, padding=1)
        self.conv4layer4no16 = nn.Conv2d(4, 3, 3, padding=1)

        '''

    def forward(self, grayscale_image):
        # apply your network's layers in the following lines:
        #x = self.conv1(grayscale_image)
        '''if decider == 0:
            x = self.batch3(grayscale_image)
        elif decider == 1:
            x = self.batch4(grayscale_image)
        elif decider == 2:
            x = self.batch5(grayscale_image)
        elif decider == 3:
            x = self.batch6(grayscale_image)
        if decider == 0:
            x = self.forward1layer2(grayscale_image)
        elif decider == 1:
            x = self.forward2_2layers(grayscale_image)
        elif decider == 2:
            x = self.forward2size3layers(grayscale_image)
        elif decider == 3:
            x = self.forward2_2size3layers(grayscale_image)
        elif decider == 4:
            x = self.forward4layers1(grayscale_image)
        elif decider == 5:
            x = self.forward4layers2(grayscale_image)
        elif decider == 6:
            x = self.forward4layers3(grayscale_image)
        elif decider == 7:
            x = self.forward4layers4(grayscale_image)
        elif decider == 8:
            x = self.forward4layers5(grayscale_image)
        elif decider == 9:
            x = self.forward4layers6(grayscale_image)
        elif decider == 10:
            x = self.forward4layers7(grayscale_image)
        elif decider == 11:
            x = self.forward4layers8(grayscale_image)
        elif decider == 12:
            x = self.forward4layers9(grayscale_image)
        elif decider == 13:
            x = self.forward4layers10(grayscale_image)
        elif decider == 14:
            x = self.forward4layers11(grayscale_image)
        elif decider == 15:
            x = self.forward4layers12(grayscale_image)
        elif decider == 16:
            x = self.forward4layers13(grayscale_image)
        elif decider == 17:
            x = self.forward4layers14(grayscale_image)
        elif decider == 18:
            x = self.forward4layers15(grayscale_image)
        elif decider == 19:
            x = self.forward4layers16(grayscale_image)
        else:
            print("ERROR OCCURED")'''
        # x = np.clip(x.cpu().detach().numpy(), -1, 1)
        x = self.Conv1(grayscale_image)
        x = F.relu(x)
        x = self.Conv2(x)
        x = torch.clamp(x, -1, 1)
        return x
    '''
    def batch3(self, grayscale_image):
        x = self.conv2size3layer1(grayscale_image)
        x = F.relu(x)
        x = self.bb8(x)
        x = self.conv2size3layer2(x)
        return x

    def batch4(self, grayscale_image):
        x = self.conv2_2size3layer1(grayscale_image)
        x = F.relu(x)
        x = self.bb8(x)
        x = self.conv2_2size3layer2(x)
        return x

    def batch5(self, grayscale_image):
        x = self.conv4layer1no6(grayscale_image)
        x = F.relu(x)
        x = self.bb8(x)
        x = self.conv4layer2no6(x)
        x = F.relu(x)
        x = self.bb2(x)
        x = self.conv4layer3no6(x)
        x = F.relu(x)
        x = self.bb8(x)
        x = self.conv4layer4no6(x)
        return x

    def batch6(self, grayscale_image):
        x = self.conv4layer1no10(grayscale_image)
        x = F.relu(x)
        x = self.bb4(x)
        x = self.conv4layer2no10(x)
        x = F.relu(x)
        x = self.bb2(x)
        x = self.conv4layer3no10(x)
        x = F.relu(x)
        x = self.bb8(x)
        x = self.conv4layer4no10(x)
        return x

    def forward1layer(self,grayscale_image):
        x = self.conv1(grayscale_image)
        return x

    def forward1layer2(self,grayscale_image):
        x =self.conv1_2(grayscale_image)
        return x

    def forward2Layers(self, grayscale_image):
        x = self.conv2layer1(grayscale_image)
        x = F.relu(x)
        x = self.conv2layer2(x)
        return x
    def forward2_2layers(self, grayscale_image):
        x = self.conv2_2layer1(grayscale_image)
        x = F.relu(x)
        x = self.conv2_2layer2(x)
        return x
    def forward2size3layers(self,grayscale_image):
        x = self.conv2size3layer1(grayscale_image)
        x = F.relu(x)
        x = self.conv2size3layer2(x)
        return x
    def forward2_2size3layers(self,grayscale_image):
        x = self.conv2_2size3layer1(grayscale_image)
        x = F.relu(x)
        x = self.conv2_2size3layer2(x)
        return x

    def forward4layers1(self,grayscale_image):
        x = self.conv4layer1no1(grayscale_image)
        x = F.relu(x)
        x = self.conv4layer2no1(x)
        x = F.relu(x)
        x = self.conv4layer3no1(x)
        x = F.relu(x)
        x = self.conv4layer4no1(x)
        return x

    def forward4layers2(self,grayscale_image):
        x = self.conv4layer1no2(grayscale_image)
        x = F.relu(x)
        x = self.conv4layer2no2(x)
        x = F.relu(x)
        x = self.conv4layer3no2(x)
        x = F.relu(x)
        x = self.conv4layer4no2(x)
        return x

    def forward4layers3(self,grayscale_image):
        x = self.conv4layer1no3(grayscale_image)
        x = F.relu(x)
        x = self.conv4layer2no3(x)
        x = F.relu(x)
        x = self.conv4layer3no3(x)
        x = F.relu(x)
        x = self.conv4layer4no3(x)
        return x

    def forward4layers4(self,grayscale_image):
        x = self.conv4layer1no4(grayscale_image)
        x = F.relu(x)
        x = self.conv4layer2no4(x)
        x = F.relu(x)
        x = self.conv4layer3no4(x)
        x = F.relu(x)
        x = self.conv4layer4no4(x)
        return x

    def forward4layers5(self, grayscale_image):
        x = self.conv4layer1no5(grayscale_image)
        x = F.relu(x)
        x = self.conv4layer2no5(x)
        x = F.relu(x)
        x = self.conv4layer3no5(x)
        x = F.relu(x)
        x = self.conv4layer4no5(x)
        return x

    def forward4layers6(self, grayscale_image):
        x = self.conv4layer1no6(grayscale_image)
        x = F.relu(x)
        x = self.conv4layer2no6(x)
        x = F.relu(x)
        x = self.conv4layer3no6(x)
        x = F.relu(x)
        x = self.conv4layer4no6(x)
        return x

    def forward4layers7(self, grayscale_image):
        x = self.conv4layer1no7(grayscale_image)
        x = F.relu(x)
        x = self.conv4layer2no7(x)
        x = F.relu(x)
        x = self.conv4layer3no7(x)
        x = F.relu(x)
        x = self.conv4layer4no7(x)
        return x

    def forward4layers8(self, grayscale_image):
        x = self.conv4layer1no8(grayscale_image)
        x = F.relu(x)
        x = self.conv4layer2no8(x)
        x = F.relu(x)
        x = self.conv4layer3no8(x)
        x = F.relu(x)
        x = self.conv4layer4no8(x)
        return x

    def forward4layers9(self, grayscale_image):
        x = self.conv4layer1no9(grayscale_image)
        x = F.relu(x)
        x = self.conv4layer2no9(x)
        x = F.relu(x)
        x = self.conv4layer3no9(x)
        x = F.relu(x)
        x = self.conv4layer4no9(x)
        return x

    def forward4layers10(self, grayscale_image):
        x = self.conv4layer1no10(grayscale_image)
        x = F.relu(x)
        x = self.conv4layer2no10(x)
        x = F.relu(x)
        x = self.conv4layer3no10(x)
        x = F.relu(x)
        x = self.conv4layer4no10(x)
        return x

    def forward4layers11(self, grayscale_image):
        x = self.conv4layer1no11(grayscale_image)
        x = F.relu(x)
        x = self.conv4layer2no11(x)
        x = F.relu(x)
        x = self.conv4layer3no11(x)
        x = F.relu(x)
        x = self.conv4layer4no11(x)
        return x

    def forward4layers12(self, grayscale_image):
        x = self.conv4layer1no12(grayscale_image)
        x = F.relu(x)
        x = self.conv4layer2no12(x)
        x = F.relu(x)
        x = self.conv4layer3no12(x)
        x = F.relu(x)
        x = self.conv4layer4no12(x)
        return x

    def forward4layers13(self, grayscale_image):
        x = self.conv4layer1no13(grayscale_image)
        x = F.relu(x)
        x = self.conv4layer2no13(x)
        x = F.relu(x)
        x = self.conv4layer3no13(x)
        x = F.relu(x)
        x = self.conv4layer4no13(x)
        return x

    def forward4layers14(self, grayscale_image):
        x = self.conv4layer1no14(grayscale_image)
        x = F.relu(x)
        x = self.conv4layer2no14(x)
        x = F.relu(x)
        x = self.conv4layer3no14(x)
        x = F.relu(x)
        x = self.conv4layer4no14(x)
        return x

    def forward4layers15(self, grayscale_image):
        x = self.conv4layer1no15(grayscale_image)
        x = F.relu(x)
        x = self.conv4layer2no15(x)
        x = F.relu(x)
        x = self.conv4layer3no15(x)
        x = F.relu(x)
        x = self.conv4layer4no15(x)
        return x

    def forward4layers16(self, grayscale_image):
        x = self.conv4layer1no16(grayscale_image)
        x = F.relu(x)
        x = self.conv4layer2no16(x)
        x = F.relu(x)
        x = self.conv4layer3no16(x)
        x = F.relu(x)
        x = self.conv4layer4no16(x)
        return x
    '''




# ---- training code -----
'''optimization_params = {"lr":[0.1, 0.01, 0.001, 0.0001],
                       "batch_size": [8,16,32],
                        "optimizer": ["Adam", "SGD"]
                       }

lr_val = 0
batch_val = 0
opt_val = 0

for i in range(3):
    if i == 0:
        hps['lr'] = 0.01
        batch_size = 32
    elif i == 1:
        hps['lr'] = 0.001
        batch_size = 8
    elif i == 2:
        hps['lr'] = 0.001
        batch_size = 16
    elif i == 3:
        hps['lr'] = 0.001
        batch_size = 32
    elif i == 4:
        hps['lr'] = 0.0001
        batch_size = 8


    for j in range(4):
        print("i = " + str(i) + " j = " + str(j))
        decider = j

        batch_size = optimization_params["batch_size"][batch_val]
        hps['lr'] = optimization_params["lr"][lr_val]
        print("printing vals")
        print(lr_val)
        print(batch_val)
        print(opt_val)'''
device = torch.device(DEVICE_ID)
print('device: ' + str(device))
net = Net().to(device=device)
criterion = nn.MSELoss()
print(net.parameters())
optimizer = optim.Adam(list(net.parameters()), lr=hps['lr'])
'''if(optimization_params["optimizer"][opt_val] == "Adam"):
    optimizer = optim.Adam(net.parameters(), lr=hps['lr'])
else:
    optimizer = optim.SGD(net.parameters(), lr=hps['lr'])'''
train_loader, val_loader = get_loaders(batch_size, device)


if LOAD_CHKPT:
    print('loading the model from the checkpoint')
    net.load_state_dict(os.path.join(LOG_DIR, 'checkpoint.pt'))

print('training begins')
final_loss = 0
final_loss_valid = 0
final_acc = 0
avg_loss = 0
avg_acc = 0
prev_loss = 1000
prev_acc = 1000
kill_flag = []
for epoch in range(max_num_epoch):
    running_loss = 0.0  # training loss of the network
    final_loss = 0
    for iteri, data in enumerate(train_loader, 0):
        inputs, targets = data  # inputs: low-resolution images, targets: high-resolution images.

        optimizer.zero_grad()  # zero the parameter gradients

        # do forward, backward, SGD step
        preds = net(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        # print loss
        running_loss += loss.item()
        print_n = 100  # feel free to change this constant
        final_loss = running_loss

        if iteri % print_n == (print_n - 1):  # print every print_n mini-batches
            print('[%d, %5d] network-loss: %.3f' %
                  (epoch + 1, iteri + 1, running_loss / 100))
            # note: you most probably want to track the progress on the validation set as well (needs to be implemented)
            #running_loss = 0.0

        if (iteri == 0) and VISUALIZE:
            hw3utils.visualize_batch(inputs, preds, targets)
    final_loss /= (5000/batch_size)
    print("Final Loss = " + str(final_loss))
    print('Saving the model, end of epoch %d' % (epoch + 1))
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    torch.save(net.state_dict(), os.path.join(LOG_DIR, 'checkpoint.pt'))
    hw3utils.visualize_batch(inputs, preds, targets, os.path.join(LOG_DIR, 'example.png'))

    if (epoch % 5 == 4):
        acc = 0
        batches = 0
        valid_loss = 0.0
        for iteri, data in enumerate(val_loader, 0):
            batches += 1
            inputs, targets = data  # inputs: low-resolution images, targets: high-resolution images.

            preds = net(inputs)
            cur_acc = np.abs(targets.cpu().detach().numpy() - preds.cpu().detach().numpy()) < (12 / 128)
            acc += cur_acc.sum() / cur_acc.size
            loss = criterion(preds, targets)

            valid_loss += loss.item()
        final_loss_valid = valid_loss/ (2000/batch_size)
        avg_loss += final_loss_valid
        print("Valid Loss printing...")
        print(valid_loss)
        print('[%d, %5d] network-loss: %.3f' %
              (epoch + 1, iteri + 1, final_loss_valid))
        acc /= batches
        final_acc = acc
        avg_acc += final_acc
        print("Current Accuracy will be printed!!!")
        print(f"{acc:.2f}/1.00")
        print("End of Validation")
        if prev_loss - final_loss_valid < 0.0001:
            print("Loss is too Low")
            kill_flag.append(1)
        else:
            kill_flag.append(0)
        n = min(len(kill_flag), 6)
        n = -n

        if sum(kill_flag[n:-1]) >= 3.5:
            break
        prev_loss = final_loss_valid
        prev_acc = final_acc
avg_loss = avg_loss/5
avg_acc = avg_acc/5
'''print("Printing to the File...")
with open("results", 'a') as f:
    f.write("Results of deneme (" + str(i) + ", " + str(j) +" )\n")
#            f.write('Values Are : lr= ' + str(optimization_params['lr'][lr_val]) + ' batch_size = ' + str(optimization_params["batch_size"][batch_val]) + " optimizer = " + str(optimization_params["optimizer"][opt_val]) + "\n")
    f.write("Final Loss = " + str(final_loss) + "\n")
    f.write("Final Acc = " + str(final_acc) + "\n")
    f.write("Final Validation Loss = " + str(final_loss_valid) + "\n")
    f.write("Average Loss = " + str(avg_loss) + "\n")
    f.write("Average ACC = " + str(avg_acc) + "\n")
    f.write("+--------------------+")
if i % 2 == 1:
    batch_val += 1
if i % 6 == 5:
    batch_val = 0
    lr_val += 1'''

print('Finished Training')


print("Estimations.npy Creating")
#net.load_state_dict(os.path.join(LOG_DIR,'checkpoint.pt'))

data_root = 'ceng483-s19-hw3-dataset'
test_set = hw3utils.HW3ImageFolder(root=os.path.join(data_root,'test'),device=device)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)



estimations2 = []
for iteri, data in enumerate(test_loader, 0):
    inputs, targets = data # inputs: low-resolution images, targets: high-resolution images.
    preds = net(inputs)
    for i in range(len(preds)):
        est = (((preds[i].permute(1,2,0).cpu().detach().numpy())/2)+0.5)*255
        estimations2.append(est)

estimations2 = np.array(estimations)
estimations2 = estimations2.reshape((2000, 80, 80, 3))
np.save("estimations.npy", estimations)



