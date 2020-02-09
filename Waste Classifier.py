#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from fastai.vision.data import *
from fastai.vision.transform import *
from fastai.vision.learner import *
from fastai.vision.learner import ClassificationInterpretation
from fastai.vision.learner import load_learner
from fastai.vision.data import DatasetType
from fastai.vision.image import open_image
from fastai.vision import models
from fastai.metrics import *
import numpy as np
import os
import zipfile as zf
from google.colab import drive
import cv2
from google.colab.patches import cv2_imshow
import re
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
import torch


# In[ ]:


drive.mount('/content/drive')


# In[ ]:


path = "/content/drive/My Drive/data/"
tfms = get_transforms(do_flip=True, flip_vert=True)
data = ImageDataBunch.from_folder(path, test="test", ds_tfms=tfms,bs=16)


# In[ ]:


data.show_batch(rows=4, figsize=(10,8))


# In[ ]:


learn = create_cnn(data,models.resnet34, metrics=error_rate)


# In[ ]:


learn.lr_find(start_lr=1e-6, end_lr=1e1)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(20,max_lr=5.13e-3)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


preds = learn.get_preds(ds_type= DatasetType.Test)


# In[ ]:


print(preds[0].shape)
preds[0]


# In[ ]:


waste_types = data.classes


# In[ ]:


max_idxs = np.asarray(np.argmax(preds[0], axis=1))


# In[ ]:


yhat = []
for max_idx in max_idxs:
  yhat.append(data.classes[max_idx])


# In[ ]:


yhat


# In[ ]:


learn.data.test_ds[0][0]


# In[ ]:


y = []

for label_path in data.test_ds.items:
  y.append(str(label_path))

pattern = re.compile("([a-z]+)[0-9]+")
for i in range(len(y)):
  y[i] = pattern.search(y[i]).group(1)


# In[ ]:


print(yhat[0:5])
print(y[0:5])


# In[ ]:


cm = confusion_matrix(y, yhat)
print(cm)


# In[ ]:


df_cm = pd.DataFrame(cm, waste_types, waste_types)

plt.figure(figsize=(10,8))
sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")


# In[ ]:


correct = 0

for r in range(len(cm)):
  for c in range(len(cm)):
    if(r==c):
      correct += cm[r,c]


# In[ ]:


accuracy = correct/sum(sum(cm))
accuracy


# In[ ]:


learn.save("/content/drive/My Drive/data/models/model3")


# In[ ]:


learn.export("/content/drive/My Drive/data/models/model3.pkl")


# In[ ]:


learn = load_learner("/content/drive/My Drive/data/models/")


# In[ ]:


learn.load("/content/drive/My Drive/data/models/model3")


# In[49]:


learn.predict(open_image("stack-of-paper.jpg"))


# In[ ]:


from PIL import Image

img = Image.open("stack-of-paper.jpg").resize((512,384), Image.ANTIALIAS)
img.save("stack-of-paper.jpg")


# In[46]:


get_ipython().system('wget "http://tmib.com/wp-content/uploads/2014/08/stack-of-paper.jpg"')


# In[ ]:


list(learn.model.parameters())[0][0]


# In[ ]:


model = torch.load("/content/drive/My Drive/data/models/model3.pth")


# In[ ]:




