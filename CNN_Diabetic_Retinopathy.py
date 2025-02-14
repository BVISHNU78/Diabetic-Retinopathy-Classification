import numpy as np
import glob,os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img , img_to_array
from sklearn.utils import resample,class_weight
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from tensorflow.keras.metrics import Precision,Recall
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import  Dropout, Conv2D, MaxPooling2D, Flatten, Dense
#import kagglehub
#path = kagglehub.dataset_download("saipavansaketh/diabetic-retinopathy-unziped")
#os.system('kaggle datasets download -d saipavansaketh/diabetic-retinopathy-unziped -p ./data')
#os.system('unzip ./datasets/diabetic-retinopathy-unziped.zip -d ./data')
#print("Dataset downloaded and unzipped to './data/'")
image_folders=glob.glob(r"D:\coding\dibatic retino\data\datas\main train\main train/*.jpeg")
csv_file=pd.read_csv(r"D:\coding\dibatic retino\trainLabels\trainLabels.csv",sep=',')
print(csv_file)
image_dir="D:\coding\dibatic retino\data\datas\main train\main train"
path_tr=glob.glob(os.path.join(image_dir,'*.jpeg'))
csv_file['Patient_id']=csv_file['image'].apply(lambda x:x.split('_')[0])
print(csv_file['Patient_id'])
csv_file['path']=csv_file['image'].apply(lambda x: os.path.join(image_dir,f"{x}.jpeg"))
print(csv_file['path'])
csv_file['exists']=csv_file['path'].apply(os.path.exists)
print(csv_file['exists'].sum(),'images found of',csv_file.shape[0],'total')
csv_file['eye']=csv_file['image'].apply(lambda x:1 if x.split('_')[-1]=='left'else 0)
print(csv_file['eye'])
num_class=csv_file['level'].nunique()
csv_file['level_cat']=csv_file['level'].map(lambda  x : to_categorical(x,num_class))
csv_file.dropna(inplace=True)
csv_file=csv_file[csv_file['exists']]
data=csv_file[['Patient_id','level']].drop_duplicates()
print(len(csv_file['Patient_id']))
labels=data['level'].values
x_train,x_vaild=train_test_split(csv_file['Patient_id'],test_size=0.2,random_state=2018)
train=csv_file[csv_file['Patient_id'].isin(x_train)]
x_vaild=csv_file[csv_file['Patient_id'].isin(x_vaild)]
print(f"train{x_train.shape[0]} validation{x_vaild.shape[0]}")
max_sample=train.groupby(['level','eye']).size().max()
balance_train=train.groupby(['level','eye'],as_index=False).apply(lambda x:resample(x,replace=True,n_samples=max_sample,random_state=42).reset_index(drop=True))
print(f"New training set size: {balance_train.shape[0]}")
print(csv_file)
balance_train[['level', 'eye']].hist(figsize=(10, 5))
plt.show()
datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
valid_datagen=ImageDataGenerator(rescale=1./255)
def generate_data_from_dataframe(df,datagen,batch_size=8,image_size=(32,32)):
    while True:
        for start in range(0,len(df),batch_size):
            batch_df=df.iloc[start:start +batch_size]
            images=[]
            labels=[]
            for _, row in batch_df.iterrows():
                try:
                    img=load_img(row['path'],target_size=image_size,color_mode="rgb")
                    img_array=img_to_array(img)/255.0
                    images.append(img_array)
                    labels.append(row['level_cat'])
                except OSError:
                    print(f"Warning: Skipping truncated image at {row['path']}")
                    continue
            if len(images) == 0:
                continue  
            images=np.array(images)
            labels=np.array(labels,dtype='float32')
            yield images, labels
        for x, y in datagen.flow(images,labels,batch_size=batch_size,shuffle=False):
            yield x, y
train_generator=generate_data_from_dataframe(train,datagen,image_size=(32, 32),batch_size=8)
valid_generator = generate_data_from_dataframe(x_vaild,valid_datagen,image_size=(32, 32),batch_size=8)

for images_batch,labels_batch in train_generator:
    break
#traning
def visualize_augment_images(generator,batch_size=8):
    k_x,k_y=next(generator)

    fig,m_axs=plt.subplots(2,4,figsize=(16,8))
    for(a_x,a_y,a_ax) in zip (k_x,k_y,m_axs.flatten()):
        a_x=np.clip(a_x*255,0,255).astype(np.uint8)
        a_ax.imshow(a_x)
        a_ax.set_title('severity{}'.format(np.argmax(a_y)))
        a_ax.axis('off')
    plt.show()
visualize_augment_images(train_generator)

#validation

def plot_valid(vaild_gen):
    v_x,v_y=next(vaild_gen)
    fig,m_axs=plt.subplots(2,4,figsize=(16,8))
    for (img,label,ax) in zip(v_x,v_y,m_axs.flatten()):
        ax.imshow(np.clip(img * 255,0,255).astype(np.uint8))
        ax.set_title('severtit:{}'.format(np.argmax(label)))
        ax.axis('off')
    plt.show()
plot_valid(valid_generator)
model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(32,32,3)))
model.add(BatchNormalization()),
model.add(MaxPooling2D((2,2))),
model.add(Dropout(0.5)),
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization()),
model.add(MaxPooling2D((2,2))),
model.add(Conv2D(128,(3,3),activation='relu')),
model.add(BatchNormalization()),
model.add(MaxPooling2D(2,2)),
model.add(Flatten()),
model.add(Dense(512,activation='relu')),
model.add(Dropout(0.5)),
model.add(Dense(5,activation='softmax')),
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(train_generator, steps_per_epoch=len(train)//32,validation_data=valid_generator,validation_steps=len(x_vaild)//32,epochs=3, batch_size=32,verbose=1) 
print(history)
validation_steps = len(x_vaild) // 32
loss,accuracy=model.evaluate(valid_generator,steps=validation_steps)
validation_steps = len(x_vaild) // 32
print("vaildation_loss",loss)
print("vaildation_accuracy",accuracy)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy',Precision(),Recall()])
model.summary()
smote = SMOTE(sampling_strategy='auto')
x_train, y_valid = smote.fit_resample(x_train, x_vaild)
base_model=VGG16(weights="imagenet",include_top=False)
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
predictions=Dense(5,activation='softmax')(x)
model=Model(inputs=base_model.input,outputs=predictions)
for layer in base_model.layers:
    layer.trainable=False
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
class_weights = {0: 1., 1: 10.}
history = model.fit(train_generator, steps_per_epoch=len(x_train)//32,validation_data=valid_generator,validation_steps=len(x_vaild)//32,epochs=3, batch_size=32,verbose=1,class_weight=class_weights)
loss,accuracy=model.evaluate(valid_generator,steps=validation_steps) 
for layer in base_model.layers:
    layer.trainable=True
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, steps_per_epoch=len(x_train)//32,validation_data=valid_generator,validation_steps=len(x_vaild)//32,epochs=3, batch_size=32,verbose=1)
loss,accuracy=model.evaluate(valid_generator,steps=validation_steps) 
all_predictions=[]
all_truelabels=[]
for x_batch,y_batch in valid_generator:
    predictions=model.predict(x_batch)
    predicted_classes=np.argmax(predictions,axis=1)
    true_classes=np.argmax(y_batch,axis=1)
    all_predictions.extend(predicted_classes)
    all_truelabels.extend(true_classes)
    if len(all_truelabels)>=len(x_train):
        break
all_predictions=np.array(all_predictions)
all_truelabels=np.array(all_truelabels)
print("confusion_matrix")
print(confusion_matrix(all_truelabels,all_predictions))
print("\nclassification_report")
print(classification_report(all_truelabels,all_predictions))
model.save("diabetics_retinopathy_model.h5")

