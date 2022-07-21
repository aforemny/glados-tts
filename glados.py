#!/usr/bin/env python

import torch
from utils.tools import prepare_text
from scipy.io.wavfile import write
import time
import sys
from subprocess import Popen, PIPE, DEVNULL
from os.path import dirname, exists

#print("Initializing TTS Engine...")

# Select the device
if torch.is_vulkan_available():
    device = 'vulkan'
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Load models
fp = dirname(sys.argv[0]) + '/../share/models/glados.pt'
if not(exists(fp)):
    fp = 'models/glados.pt'
glados = torch.jit.load(fp)

fp = dirname(sys.argv[0]) + '/../share/models/vocoder-gpu.pt'
if not(exists(fp)):
    fp = 'models/vocoder-gpu.pt'
vocoder = torch.jit.load(fp, map_location=device)

# Prepare models in RAM
for i in range(4):
    init = glados.generate_jit(prepare_text(str(i)))
    init_mel = init['mel_post'].to(device)
    init_vo = vocoder(init_mel)

text = input()

# Tokenize, clean and phonemize input text
x = prepare_text(text).to('cpu')

with torch.no_grad():

    # Generate generic TTS-output
    old_time = time.time()
    tts_output = glados.generate_jit(x)
    #print("Forward Tacotron took " + str((time.time() - old_time) * 1000) + "ms")

    # Use HiFiGAN as vocoder to make output sound like GLaDOS
    old_time = time.time()
    mel = tts_output['mel_post'].to(device)
    audio = vocoder(mel)
    #print("HiFiGAN took " + str((time.time() - old_time) * 1000) + "ms")

    # Normalize audio to fit in wav-file
    audio = audio.squeeze()
    audio = audio * 32768.0
    audio = audio.cpu().numpy().astype('int16')

    # Play audio file
    p = Popen(["aplay", "-f", "S16_LE", "-r", "22050", "-"], stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL)
    p.communicate(input=audio.tobytes())
