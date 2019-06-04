
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import squeeznet as model
import torch.nn.functional as F
import matplotlib.pyplot as plt
import configparser
from IPython import embed
import os
import pandas as pd
from PIL import Image

# In[2]:

# commented for testing config based on text file
#batch_size = 4
#epoch = 2 #it was 55 i set it to 2 just to test
#learning_rate = 0.001
#momentum = 0.9
#no_cuda = False #True calcul with GPU, False with cpu
#log_schedule = 10
#model_name =None
#want_to_test = False
#epoch_55 =False
#num_classes =9
#num_workers = 4

def readStringVar(config, varCat,varName):
    raw = config.get(varCat, varName)
    if (raw.strip().lower() == "none"):
        return None
    else:
        return raw

def readBoolVar(config, varCat,varName):
    raw = config.get(varCat, varName)
    if (raw.strip().lower() == "none"):
        return None
    else:
        if (raw.strip().lower() == "true"):
            return True
        else:
            return False

#read vars
config = configparser.ConfigParser()
config.read("config.ini")


# assign vars
batch_size = int(config.get("myvars", "batch_size"))
epoch = int(config.get("myvars", "epoch"))
learning_rate = float(config.get("myvars", "learning_rate"))
momentum = float(config.get("myvars", "momentum"))
no_cuda = readBoolVar(config,"myvars", "no_cuda") #True calcul with GPU, False with cpu
log_schedule = int(config.get("myvars", "log_schedule"))
model_name = readStringVar(config,"myvars", "model_name")
want_to_test = readBoolVar(config,"myvars", "want_to_test")
want_to_classify = readBoolVar(config,"myvars", "want_to_classify")
epoch_55 = readBoolVar(config, "myvars", "epoch_55")
num_classes = int(config.get("myvars", "num_classes"))
num_workers = int(config.get("myvars", "num_workers"))
train_data = readStringVar(config,"myvars", "train_data")
test_data = readStringVar(config,"myvars", "test_data")
to_classify_data = readStringVar(config,"myvars", "to_classify_data")

# In[3]:


data_transform = transforms.Compose([
        transforms.CenterCrop([256,256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# In[4]:


transformed_eyes = datasets.ImageFolder(root=train_data,
                                           transform=data_transform)

# In[5]:


dataset_loader = torch.utils.data.DataLoader(transformed_eyes,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers)


# In[20]:


import torchvision
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataset_loader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[transformed_eyes.classes[x] for x in classes])


# In[6]:


transformed_eyes2 = datasets.ImageFolder(root=test_data,
                                           transform=data_transform)


# In[7]:


test_loader = torch.utils.data.DataLoader(transformed_eyes2,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers)


# In[8]:


net  = model.SqueezeNet()
if model_name is not None:
    print("loading pre trained weights")
    pretrained_weights = torch.load(model_name)
    net.load_state_dict(pretrained_weights)

if no_cuda:
    net.cuda()


# In[9]:


def paramsforepoch(epoch):
    p = dict()
    regimes = [[1, 18, 5e-3, 5e-4],
               [19, 29, 1e-3, 5e-4],
               [30, 43, 5e-4, 5e-4],
               [44, 52, 1e-4, 0],
               [53, 1e8, 1e-5, 0]]
    # regimes = [[1, 18, 1e-4, 5e-4],
    #            [19, 29, 5e-5, 5e-4],
    #            [30, 43, 1e-5, 5e-4],
    #            [44, 52, 5e-6, 0],
    #            [53, 1e8, 1e-6, 0]]
    for i, row in enumerate(regimes):
        if epoch >= row[0] and epoch <= row[1]:
            p['learning_rate'] = row[2]
            p['weight_decay'] = row[3]
    return p


# In[10]:


avg_loss = list()
best_accuracy = 0.0
fig1, ax1 = plt.subplots()


# In[11]:


optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=5e-4)


# In[12]:


def adjustlrwd(params):
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = params['learning_rate']
        param_group['weight_decay'] = params['weight_decay']


# In[13]:


def train(epoch):

    # set the optimizer for this epoch
    if epoch_55:
        params = paramsforepoch(epoch)
        print("Configuring optimizer with lr={:.5f} and weight_decay={:.4f}".format(params['learning_rate'], params['weight_decay']))
        adjustlrwd(params)
    ###########################################################################

    global avg_loss
    correct = 0
    net.train()
    for b_idx, (data, classes) in enumerate(dataset_loader):
        # trying to overfit a small data
        # if b_idx == 100:
        #     break

        if no_cuda:
            data, classes = data.cuda(), classes.cuda()
        # convert the data and targets into Variable and cuda form
        data, classes = Variable(data), Variable(classes)

        # train the network
        optimizer.zero_grad()
        scores = net.forward(data)
        print(scores.shape)
        scores = scores.view(data.size()[0], num_classes)
        loss = F.nll_loss(scores, classes)
        # compute the accuracy
        pred = scores.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(classes.data).cpu().sum()

        avg_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if b_idx % log_schedule == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (b_idx+1) * len(data), len(dataset_loader.dataset),
                100. * (b_idx+1)*len(data) / len(dataset_loader.dataset), loss.item()))

            # also plot the loss, it should go down exponentially at some point
            ax1.plot(avg_loss)
            fig1.savefig("Squeezenet_loss.jpg")

    # now that the epoch is completed plot the accuracy
    train_accuracy = correct / float(len(dataset_loader.dataset))
    print("training accuracy ({:.2f}%)".format(100*train_accuracy))
    return (train_accuracy*100.0)


# In[14]:


def val(i):
    global best_accuracy
    correct = 0
    net.eval()
    for idx, (data, target) in enumerate(test_loader):
        if idx == 73:
            break

        if no_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        # do the forward pass
        score = net.forward(data)
        pred = score.data.max(1)[1] # got the indices of the maximum, match them
        correct += pred.eq(target.data).cpu().sum()

    print("predicted {} out of {}".format(correct, 73*64))
    val_accuracy = correct / (73.0*64.0) * 100
    print("accuracy = {:.2f}".format(val_accuracy))

    # now save the model if it has better accuracy than the best model seen so forward
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        # save the model
    torch.save(net.state_dict(),'bsqueezenet_onfulldatatest'+str(i)+'.pth')
    return val_accuracy


# In[15]:


def test():
    # load the best saved model
    weights = torch.load('bsqueezenet_onfulldatatest'+str(epoch)+'.pth')
    net.load_state_dict(weights)
    net.eval()

    test_correct = 0
    total_examples = 0
    accuracy = 0.0
    for idx, (data, classes) in enumerate(test_loader):
        total_examples += len(classes)
        data, classes = Variable(data), Variable(classes)
        if no_cuda:
            data, classes = data.cuda(), classes.cuda()

        scores = net(data)
        pred = scores.data.max(1)[1]
        test_correct += pred.eq(classes.data).cpu().sum()
    print("Predicted {} out of {} correctly".format(test_correct, total_examples))
    return 100.0 * test_correct / (float(total_examples))
def classifythis():
    #loading model 
    net  = model.SqueezeNet()
    weights = torch.load('bsqueezenet_onfulldata.pth')
    net.load_state_dict(weights)
    net.eval()
    #loading classes    
    target = ['BottomCenter','BottomLeft','BottomRight','MiddleCenter','MiddleLeft','MiddleRight','TopCenter','TopLeft', 'TopRight']
    #creating var that will contain the result
    df = pd.DataFrame(columns=["file", "prediction"])
    for file in os.listdir('data/toClassify'):
        if file.endswith(".jpg"):
            image = Image.open("data/toClassify/" + str(file))
            data_transform = transforms.Compose([
            transforms.CenterCrop([256,256]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
            image_tensor = data_transform(image).float()
            image_tensor = image_tensor.unsqueeze_(0)
            input = Variable(image_tensor)
            output = net.forward(input)
            result = target[output.data.numpy().argmax()]
            df = df.append({"file": file,"prediction":  result}, ignore_index=True)
    df.to_csv("result.csv",header=True,index=None,sep=",")

# In[16]:


def main():
    if want_to_classify:

    if not want_to_test:
        fig2, ax2 = plt.subplots()
        train_acc, val_acc = list(), list()
        for i in range(1,epoch+1):
            train_acc.append(train(i))
            val_acc.append(val(i))
            ax2.plot(train_acc, 'g')
            ax2.plot(val_acc, 'b')
            fig2.savefig('train_val_accuracy.jpg')
    else:
        test_acc = test()
        print("Testing accuracy on CIFAR-10 data is {:.2f}%".format(test_acc))


# In[17]:


main()



