


import sys
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
import alexnet as model2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import configparser
from IPython import embed
import os
import pandas as pd
from PIL import Image


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
model_num = int(config.get("myvars", "model_num")) #0 SqueezNet 1 AlexNet
log_schedule = int(config.get("myvars", "log_schedule"))
model_name = readStringVar(config,"myvars", "model_name")
epoch_55 = readBoolVar(config, "myvars", "epoch_55")
num_classes = int(config.get("myvars", "num_classes"))
num_workers = int(config.get("myvars", "num_workers"))
train_data = readStringVar(config,"myvars", "train_data")
test_data = readStringVar(config,"myvars", "test_data")
to_classify_data = readStringVar(config,"myvars", "to_classify_data")
action = readStringVar(config,"myvars", "action")






if(action.strip().lower()=="test"):
    want_to_test = True
    want_to_classify = False
    want_to_show = False
else:
    if(action.strip().lower()=="train"):
        want_to_test = False
        want_to_show = False
        want_to_classify = False
    else:
        if(action.strip().lower()=="classify"):
            want_to_test = False
            want_to_classify = True
            want_to_show = False
        else:
            if (action.strip().lower() == "show_weights"):
                want_to_show = True
                want_to_classify = False
                want_to_test = False
            else:
                print("Wrong configuration of the action parameter which can be either train or test or classify")
                sys.exit()
# In[3]:


#transform data while reading them
#take juste the center part where the eye is positioned
data_transform = transforms.Compose([
        transforms.CenterCrop([256,256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# In[4]:

#specify where the data is find and the transformation
transformed_eyes = datasets.ImageFolder(root=train_data,
                                           transform=data_transform)

# In[5]:

#load daya in batch and apply transformation
dataset_loader = torch.utils.data.DataLoader(transformed_eyes,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers)


# In[20]:


import torchvision
# to show some image sample
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

#the same with test data
transformed_eyes2 = datasets.ImageFolder(root=test_data,
                                           transform=data_transform)


# In[7]:


test_loader = torch.utils.data.DataLoader(transformed_eyes2,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers)


# In[8]:
print("--> Configuration can be done via the file Config.ini")
if model_num == 0 :
    print("****** model architechture type: Squeezenet ****** ")
    net  = model.SqueezeNet()
if model_num == 1 :
    print("****** model architechture type: AlexNet *******")
    net =  model2.AlexNet()
else:
    net = model2.AlexNet()


if model_name is not None:
    print("loading pre trained weights")
    pretrained_weights = torch.load(model_name)
    net.load_state_dict(pretrained_weights)

if no_cuda:
    net.cuda()


# In[9]:

#specify some parameters for the training
def paramsforepoch(epoch):
    p = dict()
    regimes = [[1, 18, 5e-3, 5e-4],
               [19, 29, 1e-3, 5e-4],
               [30, 43, 5e-4, 5e-4],
               [44, 52, 1e-4, 0],
               [53, 1e8, 1e-5, 0]]
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

criterion = nn.CrossEntropyLoss()
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
        if no_cuda:
            data, classes = data.cuda(), classes.cuda()
        # convert the data and targets into Variable and cuda form
        data, classes = Variable(data), Variable(classes)

        # train the network
        optimizer.zero_grad()
        #scores contain the result of NN
        #a vector of 9 probability
        scores = net.forward(data)
        #view is to reshape the vector to the right form
        scores = scores.view(data.size()[0], num_classes)
        _, prediction = torch.max(scores.data, 1)
        correct += torch.sum(prediction == classes.data).float()
        loss = criterion(scores, classes)
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
    train_accuracy = correct / len(dataset_loader.dataset) *100

    print("training accuracy ({:.2f}%)".format(train_accuracy))
    return (train_accuracy)


# In[14]:


def val():
    global best_accuracy
    correct = 0 #otherwise he will not save the model
    net.eval()
    test = 0
    for idx, (data, target) in enumerate(test_loader):
        if no_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        # do the forward pass
        score = net.forward(data)
        pred = score.data.max(1)[1] # got the indices of the maximum, match them
        print("pred")
        print(pred)
        test += torch.sum(pred == target.data).float()
        print("correct")
        print(correct)
    print("predicted {} out of {}".format(test, len(test_loader.dataset)))
    val_accuracy = test / len(test_loader.dataset) *100
    print(val_accuracy)
    print("accuracy = {}".format(val_accuracy))

    # now save the model if it has better accuracy than the best model seen so forward
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        if model_num == 0:
            torch.save(net.state_dict(),'bsqueezenet_onfulldatatest.pth')
        if model_num == 1:
            torch.save(net.state_dict(),'tryAlexNet.pth')
    return val_accuracy


# In[15]:


def test():
    # load the best saved model
    if model_num == 0:
        weights = torch.load('bsqueezenet_onfulldatatest.pth')
    if model_num == 1:
        weights = torch.load('tryAlexNet.pth')
    
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
    print("classifythis")
    if model_num == 0:
        net  = model.SqueezeNet()
        weights = torch.load('bsqueezenet_onfulldatatest.pth')
    if model_num == 1:
        net = model2.AlexNet()
        weights = torch.load('tryAlexNet.pth')
    
    print("model loded")
    net.load_state_dict(weights)
    print("weights loded")
    # some layer react differently when they are in training and in production
    # net.train() is to let know the model he is in train mode
    # net.eval() when we need the result of the model
    net.eval()
    #loading classes
    target = ['BottomCenter','BottomLeft','BottomRight','MiddleCenter','MiddleLeft','MiddleRight','TopCenter','TopLeft', 'TopRight']
    #creating var that will contain the result
    df = pd.DataFrame(columns=["file", "prediction","proba_class0","proba_class1","proba_class2","proba_class3","proba_class4","proba_class5","proba_class6","proba_class7","proba_class8"])
    i = 0
    number_of_file = len(os.listdir(to_classify_data))
    for file in os.listdir(to_classify_data):
        if file.lower().endswith(".jpg"):
            image = Image.open(to_classify_data+ "/" + str(file))
            data_transform = transforms.Compose([
            transforms.CenterCrop([256,256]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
            image_tensor = data_transform(image).float()
            image_tensor = image_tensor.unsqueeze_(0)
            input = Variable(image_tensor)
            output = net.forward(input)
            output = output.detach().numpy()
            print("probabilitie of the image for each class :")
            print(output)
            result = target[output.argmax()]
            print("the class with the highest proba :")
            print (result)
            df = df.append({"file": file,"prediction":  result,"proba_class0": output[0][0],"proba_class1": output[0][1],"proba_class2": output[0][2],"proba_class3": output[0][3],"proba_class4": output[0][4],"proba_class5": output[0][5],"proba_class6": output[0][6],"proba_class7": output[0][7],"proba_class8": output[0][8]}, ignore_index=True)
            print(str(i) + " of " + str(number_of_file) + " done")
            i = i+1
    df.to_csv("result.csv",header=True,index=None,sep=",")

# In[16]:
def show_Weights(summary):
    print("charging weights")
    if model_num == 0:
        net = model.SqueezeNet()
        weights = torch.load('bsqueezenet_onfulldatatest.pth')
    if model_num == 1:
        net = model2.AlexNet()
        weights = torch.load('tryAlexNet.pth')
    print("number of layer = ")
    print(len(weights)/2)
    for layer_name in (weights):
        print(layer_name)
        if summary == "yes":
            print(len(weights[layer_name]))
        else:
            print(weights[layer_name])
if __name__== '__main__':
    if want_to_show:
        torch.set_printoptions(profile="full")
        summary = input("show summary ? yes/no ")
        show_Weights(summary.lower())

    else:
        if want_to_classify:
            print("You want to classify the immages using exiting trained model")
            classifythis()
        else:
            print("You want train the model using data in " + train_data)
            if not want_to_test:
                fig2, ax2 = plt.subplots()
                train_acc, val_acc = list(), list()
                for i in range(1,epoch+1):
                    train_acc.append(train(i))
                    val_acc.append(val())
                    ax2.plot(train_acc, 'g')
                    ax2.plot(val_acc, 'b')
                    fig2.savefig('train_val_accuracy.jpg')
            else:
                print("You want test the accuracy of the predection of model using data in " + test_data)
                test_acc = test()




