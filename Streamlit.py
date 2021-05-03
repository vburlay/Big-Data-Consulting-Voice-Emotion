import streamlit as st
import os, urllib, cv2
import tensorflow as tf
import torch
import keras
import torch.nn as nn
from pathlib import Path
from torchvision import models
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd
import numpy as np
import librosa
import librosa.display
from PIL import Image
import os, urllib
from torchvision import transforms
from scipy import signal
from scipy.io import wavfile
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
st.write("""
# Stimmungserkennung App
*Vladimir Burlay*
""")
df = pd.DataFrame({
  'first column': ['Keras', 'PyTorch'], 
    })

class_names = ['neutral','calm', 'happy', 'sad', 'angry', 'fear', 'disgust','surprise'] 

def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/MDK1192/Big-Data-Consulting-Voice-Emotion/Dev_Branch_Vladimir/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

########Keras###############################################
batch_size = 64
img_width, img_height, img_num_channels = 3,200,300
loss_function = sparse_categorical_crossentropy
no_classes = 8
optimizer = Adam()
verbosity = 1
input_shape = (img_width, img_height, img_num_channels)

DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding="SAME", use_bias=False)

class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
    
def build_model():
    model = keras.models.Sequential()
    model.add(DefaultConv2D(64, kernel_size=3, strides=1,
                        input_shape= input_shape ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))
    prev_filters = 8
    for filters in [64] * 1 + [128] * 1 + [256] * 1: 
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters, strides=strides))
        prev_filters = filters
    model.add(keras.layers.GlobalAvgPool2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(no_classes, activation="softmax"))
    optimizer = keras.optimizers.Nadam(lr=1e-3)
    model.compile(loss=loss_function,
              optimizer=optimizer,  metrics=['accuracy'])
    return model


model_prod = build_model()
model_prod.load_weights('path_model.h5')
########Pytorch#############################################
def model_res():
    device = "cpu"
    spec_resnet = models.resnet50(pretrained=True)

    for param in spec_resnet.parameters():
        param.requires_grad = False

        spec_resnet.fc = nn.Sequential(nn.Linear(spec_resnet.fc.in_features,500),
                               nn.ReLU(),
                               nn.Dropout(), nn.Linear(500,8))
        return spec_resnet

img_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

spec_resnet_prod = model_res()

PATH_TO_RAVDESS = Path.cwd()/ 'spec_resnet_temp.pth'
spec_resnet_prod.load_state_dict(torch.load(PATH_TO_RAVDESS))  
spec_resnet_prod.eval()
################################################################

readme_text = st.markdown(get_file_content_as_string("Anweisungen.md"))

                          
st.sidebar.title("Was ist zu tun")
app_mode = st.sidebar.selectbox("Wählen Sie den App-Modus",
        ["Anweisungen anzeigen","Ausführen der App", "Den Quellcode anzeigen"])


if app_mode == "Ausführen der App":
       uploaded_files = st.file_uploader("Wählen Sie bitte eine wav/png-Datei", accept_multiple_files=True)
       readme_text.empty()
       for uploaded_file in uploaded_files:
           img = Image.open(uploaded_file).convert('RGB')
           img_tensor = img_transforms(img).unsqueeze(0)
           prediction =  spec_resnet_prod(img_tensor)
           predicted_class = class_names[torch.argmax(prediction)]
           st.image(uploaded_file, caption='Spektogramm')
           st.write(predicted_class)
elif app_mode == "Anweisungen anzeigen":
    st.sidebar.success('Kommen Sie bitte zur App')
elif app_mode == "Den Quellcode anzeigen":
    readme_text.empty()
    st.code(get_file_content_as_string("Streamlit.py"))
    
    
#        bytes_data = uploaded_file.read()

#        plt.gcf().savefig("{}{}_{}.png".format(filename.parent,dpi,filename.name), dpi=dpi)

#    if app_mode == "Keras":
#          audio_tensor, sr = librosa.load(uploaded_file, sr=None)
#          spectrogram = librosa.feature.melspectrogram(audio_tensor, sr=sr)
#          log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
#          librosa.display.specshow(log_spectrogram, sr=sr, x_axis='time', y_axis='mel')
#          img = Image.open(uploaded_file).convert('RGB')
#          img_tensor = img_transforms(img).unsqueeze(0)
#          img_dat= (tf.cast(img_tensor, tf.float32) / 255).numpy()
#          prediction = model_prod.predict(img_dat)
#         predicted_class = class_names[prediction]
#         st.write(predicted_class)
         

       
