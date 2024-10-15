import argparse 
import os
import torch
import torchaudio
import numpy as np
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from wav2vec2_vib import Model
from torchinfo import summary
import librosa

# device = "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(
    device=device,
    ssl_cpkt_path='xlsr2_300m.pt', # https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/xlsr/README.md#:~:text=XLS%2DR%20300M-,download,-XLS%2DR%201B
).to(device)
# 가중치 로드 시 에러 핸들링 추가
try:
    model.load_state_dict(torch.load('vib_conf-5_gelu_2s_may27_epoch6.pth', map_location=device))
    print("Model weights loaded successfully!")
except FileNotFoundError:
    print("Model weights file not found. Please check the file path.")
except Exception as e:
    print(f"Error loading model weights: {e}")
    
# activation_layer = [model.ssl_model.model.feature_extractor.conv_layers[-1][0]]
# gradient_layers = model.backend.m_utt_level
target_layer = [model.ssl_model.model.feature_extractor.conv_layers[-1][0]]

# 오디오 로드
audio_path = "/home/woonj/grad-cam/1조인성_수상소감.wav"

waveform, sr = librosa.load(audio_path, sr=16000)

waveform =  torch.Tensor(waveform)
waveform = waveform.unsqueeze(0)  # 배치 차원 추가 후 디바이스로 이동

input_tensor = waveform.to(device)

# Grad-CAM 초기화
cam = GradCAM(model=model, target_layers=target_layer, reshape_transform=None)

grayscale_cam = cam(input_tensor=input_tensor, targets=None)