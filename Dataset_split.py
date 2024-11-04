import pandas  as pd
import os
import sys
import shutil

labels=pd.read_csv(r"D:\coding\dibatic retino\trainLabels\trainLabels.csv")
data=r"D:\coding\dibatic retino\data\datas"
DR=r"D:\coding\dibatic retino\data\DR"
if not os.path.exists(DR):
    os.mkdir(DR)
for filename,class_name in labels.values:
    if not os.path.exists(DR +str(class_name)):
        os.mkdir(DR + str(class_name))
    src_path = data +'/'+filename +'.jpeg'
    dst_path=DR +str(class_name) + '/'+ filename + '.jpeg'
    try:
        shutil.copy(src_path,dst_path)
    except IOError as e:
        print("Unable to copy file {} to {}".format(src_path,dst_path))
    except:
       print('When try copy file{} to {}, unexpected error:{}'.format(src_path,dst_path,sys.exc_info()))
