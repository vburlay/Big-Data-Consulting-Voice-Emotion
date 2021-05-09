import streamlit as st
import os, urllib, cv2
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import models
from functools import partial
import pandas as pd
import numpy as np
from PIL import Image
import os, urllib
from torchvision import transforms


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


       
