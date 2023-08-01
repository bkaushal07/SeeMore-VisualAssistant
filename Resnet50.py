 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models.segmentation import fcn_resnet50

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# In[2]:


class ToIntTensor(transforms.ToTensor):
    """A custom transform that replaces "ToTensor". ToTensor always converts to a 
    a float in range [0,1]. This one converts to an integer, which can represent
    our class labels per pixel in an image segmentation problem"""
    def __call__(self, pic):
        tensor = super().__call__(pic)
        tensor = (tensor * 255).to(torch.int64)
        return tensor

def get_voc_dataloader(batch_size=4):
    """Get the VOC 2007 segmentation dataset and return PyTorch 
    dataloaders for both training and validation. 
    """
    image_transforms = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),
                                           transforms.Normalize(mean=[0, 0, 0],std=[1,1,1])]) # TODO
    label_transforms = transforms.Compose([transforms.Resize((64,64)),ToIntTensor()])# TODO

    # This downloads the data automatically and creates a "dataset" object that applies the transforms
    data_dir = "C:/Users/adisr"  
    train_dataset = datasets.VOCSegmentation(data_dir, year='2007', image_set='train', download=True, transform=image_transforms, target_transform=label_transforms)
    val_dataset = datasets.VOCSegmentation(data_dir, year='2007', image_set='val', download=True, transform=image_transforms, target_transform=label_transforms)

    # Create data loaders for the datasets - necessary for efficient training
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dl, val_dl
    
    


# In[7]:


def train_epoch(model, train_dl, val_dl, optimizer, device):
    """
    Train one epoch of model with optimizer, using data from train_dl.
    Do training on "device". 
    Return the train and validation loss and validation accuracy.
    """
    # We'll use the cross entropy loss. There's a nice feature that it
    # allows you to "ignore_index". In this case index 255 is the mask to ignore
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # recommend to use in constructing loss
    
    train_loss = 0
    val_loss = 0
    accuracy = 0
    
    
    model.train()
    for x_train,y_train in train_dl:
        x_train,y_train = x_train.to(device).float(),y_train.to(device).long()
        optimizer.zero_grad()
        outputs = model(x_train)
        outputs = outputs['out']
        labels = torch.argmax(outputs,dim = 1)
        loss = criterion(outputs,y_train.squeeze(1))
        loss.backward() 
        optimizer.step()
        train_loss = loss.item()/len(train_dl)
        
    
    model.eval()
    with torch.no_grad():
        for x_val,y_val in val_dl:
            x_val, y_val = x_val.to(device).float(), y_val.to(device).long()
            y = torch.argmax(y_val,dim=1)
            val_outputs = model(x_val)
            val_outputs = val_outputs['out']
            val_labels = torch.argmax(val_outputs,dim=1)
            val_loss += criterion(val_outputs,y_val.squeeze(1))
            val_loss = val_loss.item()/len(val_dl)
            accuracy = (val_labels == y).float().mean()
    return train_loss, val_loss, accuracy

        


# ### Main loop

# In[8]:


torch.cuda.empty_cache()


# In[9]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define some hyperparameters
batch_size = 32  # Adjust batch size to make maximal use of GPU without running out of memory 
epochs = 50
learning_rate = 0.01
n_class = 21  # The class labels are 0...20. The label "255" is interpreted as a "mask" meant to be ignored

# Load model and data
model = fcn_resnet50(n_class=n_class).to(device)
train_dl, val_dl = get_voc_dataloader(batch_size=batch_size)

# Training loop
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.)

train_losses, val_losses, acc= [], [], []
for epoch in range(epochs):
    train_loss, val_loss,accuracy = train_epoch(model, train_dl, val_dl, optimizer, device)
    
    # Print the loss, and store for plotting
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    acc.append(accuracy)
    print('Epoch %d: Train loss: %.3f | Val loss: %.3f | Acc: %.3f' % (epoch+1, train_loss, val_loss,accuracy))


# ## Post training visualization and analysis

# In[10]:


cmap = plt.cm.get_cmap('tab20', n_class + 1)  # tab20 is a colormap with 20 distinct colors
examplex, exampley = next(iter(val_dl))
examplex, exampley = examplex.float().to(device), exampley.long().to(device)
output = model(examplex)['out']

output = output.cpu().argmax(dim=1).detach().numpy()
examplex = examplex.cpu()
exampley = exampley.cpu()


example_indexs = [1,3,5,7]
for example_index in example_indexs:
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].imshow(examplex[example_index].permute(1, 2, 0))
    axes[0].set_title('Original Image')
    axes[1].imshow(exampley[example_index].permute(1, 2, 0))
    axes[1].set_title('True Segmentation')
    axes[2].imshow(output[example_index],cmap = cmap)
    axes[2].set_title('Predicted Segmentation')
    plt.show()

plt.plot(train_losses)
plt.plot(val_losses)
plt.xlabel("Iterations")
plt.ylabel("Losses")
plt.legend(["Training Loss","Validation Loss"])
plt.show()



# In[11]:


model.eval()
with torch.no_grad():
    all_acc = []
    all_conf = []
    for i, (inputs, labels) in enumerate(val_dl):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)['out']
        labels = labels.squeeze(1).flatten(start_dim=1)  # batch, h, w  - integer values 0..20 or 255 for mask
        predicted_label = outputs.argmax(axis=1).flatten(start_dim=1)  # batch, h, w, integer 0...20  
        probs = outputs.softmax(axis=1)  # batch, n_class, h, w
        confidence = probs.max(axis=1).values.flatten(start_dim=1)  # Confidence in predicted label
        accuracy = (predicted_label == labels)
        accuracy_filter = accuracy[labels < 255]
        confidence_filter = confidence[labels < 255]
        all_acc.append(accuracy_filter)
        all_conf.append(confidence_filter)
        
all_acc = torch.cat(all_acc).cpu().numpy()  # accuracy to predict pixel class across all pixels and images, excluding masks
all_conf = torch.cat(all_conf).cpu().numpy()  # confidence of prediction for each pixel and image, excluding masks
        
# Get the average confidence and accuracy for points within different confidence ranges
bins = 10
bin_boundaries = np.linspace(0, 1, bins + 1)
bin_lowers = bin_boundaries[:-1]
bin_uppers = bin_boundaries[1:]
bin_centers = 0.5*(bin_lowers+bin_uppers)
bin_acc = np.zeros(bins)  # Store accuracy within each bin
bin_conf = np.zeros(bins)  # Store confidence within each bin
bin_frac = np.zeros(bins)  # Store the fraction of data in included in each bin
for i in range(bins):
    in_bin = np.logical_and(all_conf >= bin_lowers[i], all_conf < bin_uppers[i])
    bin_frac[i] = np.sum(in_bin) / len(all_conf)  # fraction of points in bin
    if bin_frac[i] > 0.:
        bin_acc[i] = all_acc[in_bin].mean()  # average accuracy in this bin
        bin_conf[i] = all_conf[in_bin].mean()  # average confidence in this bin
    else:
        bin_acc[i], bin_conf[i] = 0, 0  # If no points are in this bin, they don't contribute to ECE anyway


# In[16]:


plt.plot(bin_conf,bin_acc)
plt.xlabel("Confidence")
plt.ylabel("Accuracy")
plt.title("Confidence Calibration Curve")
plt.show()
ece = np.sum(bin_frac * abs(bin_conf - bin_acc)) # TODO
print("ECE: %.3f" %ece)

plt.bar(bin_conf,bin_acc)
plt.xlabel('Confidence')
plt.ylabel('Accuracy')
plt.title('confidence versus accuracy bar chart')
plt.show()


# In[ ]:




