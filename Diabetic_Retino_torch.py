from cv2 import transform
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas  as pd
import sys,os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch, torchvision, timeit,copy
from torch.utils.data.dataloader import DataLoader
print(torch.cuda.is_available())
path=r"D:\coding\dibatic retino\data\dia"
classes=os.listdir(os.path.join(path,'train'))
num_classes=len(classes)
print(f"Number of classes : {num_classes}\nClasses : {classes}")
eye=os.listdir(os.path.join(path,'train','DR0'))
img = np.asarray(Image.open(os.path.join(path, 'train', 'DR0', eye[0])))
print(f"Image shape : {img.shape}")
plt.imshow(img)
plt.show()
samples=[]
for i in classes:
    images =os.listdir(os.path.join(path,'train',i))
    for j in images[:5]:
        samples.append(np.asarray(Image.open(os.path.join(path,'train',i,j))))
fig,axes =plt.subplots(ncols=5,nrows=5,figsize=(10,10))
idx=0
for i in range(num_classes):
    print(f'eyeRow{i+1}:{classes[i]}')
    for j in range(5):
        axes[i,j].imshow(samples[idx])
        idx+=1
plt.show()
transform =torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.24,0.225]),
])
train=torchvision.datasets.ImageFolder(os.path.join(path,'train'),transform=transform)
val=torchvision.datasets.ImageFolder(os.path.join(path,'test'),transform=transform)
batch_size=64
train_dl=DataLoader(train,batch_size=batch_size,shuffle=True)
val_dl=DataLoader(val,batch_size=batch_size,shuffle=False)
train=None
val=None
print(f"Number of train batches:{len(train_dl)}\nNumber of Test batches:{len(val_dl)}")
device='cuda' if torch.cuda.is_available() else 'cpu'
print(f" Avaliable device :{device}")
model=torchvision.models.resnet18(weights='IMAGENET1K_V1')
model.fc=torch.nn.Linear(in_features=512,out_features=num_classes,bias=True)
model=model.to(device)
start=timeit.default_timer()
total=0
correct=0
with torch.no_grad():
    for (img,label) in train_dl:
        img,label=img.to(device),label.to(device)
        output=model(img)
        _,Predicated=torch.max(output.data,1)
        total+=label.size(0)
        correct+=(Predicated==label).sum().item()
        torch.cuda.empty_cache()
end=timeit.default_timer()
print(f"Accuarcy on Train set:{(correct/total)*100}\tExecution time :{end-start}seconds")
start=timeit.default_timer()
total=0
correct=0
with torch.no_grad():
    for (img,label) in val_dl:
        img,label=img.to(device),label.to(device)
        output=model(img)
        _,Predicated=torch.max(output.data,1)
        total+=label.size(0)
        correct+=(Predicated==label).sum().item()
        torch.cuda.empty_cache()
end=timeit.default_timer()
print(f"Accuarcy on vaildation set:{(correct/total)*100}\tExecution time :{end-start}seconds")
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
xent=torch.nn.CrossEntropyLoss()
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)
if device!='cpu':
    model=model.to(device)

val_loss=[]
val_acc=[]
train_loss=[]
train_acc=[]
epochs=30
best_accuracy=0
best_model=copy.deepcopy(model.state_dict())
start=timeit.default_timer()
for epoch in range(1,epochs+1):
    print(f"Epoch:{epoch}\n")
    dataloader=None

for phase in['train','val']:
    loss=0
    correct=0
    total=0
    batch_num=0

if phase == 'train':
    model.train()
    dataloader=train_dl
else:
    model.eval()
    dataloader=val_dl
for (img,label) in dataloader:
    img,label =img.to(device),label.to(device)
    optimizer.zero_grad()

with torch.set_grad_enabled(phase =='train'):
    output=model(img)
    _,pred=torch.max(output.data,1)
    loss=xent(output,label)
    if phase=='train':
        loss.backward()
        optimizer.step()
loss+=loss.item()*img.size(0)
torch.cuda.empty_cache()
correct +=torch.sum(pred ==label.data)
total+=label.size(0)
batch_num+=1
if batch_num%100==0:
    print(f"Epoch:{epoch}\t{phase}batch{batch_num}completed")
    if phase =="train":
        train_loss.append(loss/len(train_dl))
        train_acc.append(correct/total)
        scheduler.step()
        print(f'{phase} Loss: {train_loss[-1]:.4f}\tAccuracy: {train_acc[-1]:.4f}')
    else :
        val_loss.append(loss/len(val_dl))
        val_acc.append(correct/total)
        print(f'{phase} Loss: {val_loss[-1]:.4f}\tAccuracy: {val_acc[-1]:.4f}')

    if val_acc[-1] > best_accuracy :
        best_accuracy = val_acc[-1]
        best_model = copy.deepcopy(model.state_dict())

        torch.cuda.empty_cache()
    
    print()
model.eval()
end = timeit.default_timer()
print(f"Total time elapsed = {end - start} seconds")
name = f'DR_pytorch_{epoch}_{best_accuracy:.5f}.pth'
model.load_state_dict(best_model)
model = model.to('cpu')
torch.save(model.state_dict(), name)
train_loss = [i.item() for i in train_loss]
val_loss = [i.item() for i in val_loss]
train_acc = [i.item() for i in train_acc]
val_acc = [i.item() for i in val_acc]
print(f"Length of train_acc: {len(train_acc)}")
print(f"Length of val_acc: {len(val_acc)}")
print(f"Expected length (epochs): {epochs}")
plt.figure()
#plt.plot(range(1,epochs+1), train_loss, color='blue')
#plt.plot(range(1,epochs+1), val_loss, color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend(['Train Loss', 'Validation Loss'])
plt.show()
plt.figure()
plt.plot(range(1,epochs+1), train_acc, color='blue')
plt.plot(range(1,epochs+1), val_acc, color='red')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend(['Train Accuracy', 'Validation Accuracy'])
plt.show()