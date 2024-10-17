import argparse 
import os
import torch
import torchaudio
import numpy as np
from wav2vec2_vib import Model
from torchinfo import summary
import librosa
from audio_cam import AudioGradCAM
from tqdm import tqdm  # Import tqdm

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = Model(
    device=device,
    ssl_cpkt_path='xlsr2_300m.pt', # Update this path as needed
).to(device)

# Load model weights with error handling
try:
    model.load_state_dict(torch.load('/datad/pretrained/AudioDeepfakeCMs/vib/vib_conf-5_gelu_acmccs_apr3_moreko_telephone_epoch22.pth', map_location=device))
    print("Model weights loaded successfully!")
except FileNotFoundError:
    print("Model weights file not found. Please check the file path.")
    exit()  # Exit if model weights are not found
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit()  # Exit on other errors

# Specify the directory containing the audio files
audio_directory = "/home/woonj/analyze_missclassification/cnsl_real_fake_audio/2-DSD-corpus/Real/inthewild_bona"

# Initialize variables to count negative CAM values
total_files = 0
negative_cam_count = 0

# Setup Grad-CAM
target_layer = model.LL
grad_cam = AudioGradCAM(model=model, target_layers=[target_layer])

# Get the list of audio files
audio_files = os.listdir(audio_directory)


# Open the output file for writing
output_file = "cam_analysis_results.txt"

with open(output_file, "w") as f:
    # Write the header to the output file
    f.write("Filename\tPositive Count\tNegative Count\tNegative Ratio\n")

    # Loop through all audio files in the specified directory with tqdm progress bar
    for filename in tqdm(os.listdir(audio_directory), desc="Processing audio files", unit="file"):
        audio_path = os.path.join(audio_directory, filename)
        try:
            # Load the audio file
            waveform, sr = librosa.load(audio_path, sr=16000)
            
            # Prepare waveform tensor
            waveform = torch.Tensor(waveform).unsqueeze(0)  # Add batch dimension
            
            # Compute CAM values
            cam = grad_cam(waveform)

            # Count positive and negative CAM values
            positive_count = (cam > 0).sum().item()
            negative_count = (cam <= 0).sum().item()

            # Calculate the ratio of negative values
            total_values = positive_count + negative_count
            negative_ratio = negative_count / total_values if total_values > 0 else 0
            # Write the results for this audio file immediately
            f.write(f"{filename}\t{positive_count}\t{negative_count}\t{negative_ratio:.4f}\n")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Print summary of results
print(f"Analysis results saved to {output_file}.")
