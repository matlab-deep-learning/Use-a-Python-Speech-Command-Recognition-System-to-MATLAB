from speechCommandModel import *
import numpy as np
import librosa

# Load pretrained model
model = CNN()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Read audio
sr = 16e3
samples, sr = librosa.load(filename, sr)

# Define Mel spectrogram arguments
args = {
  "n_fft": 512,
  "hop_length": 160,
  "win_length":512,
  "window": "hann",
  "center":False,
  "n_mels": 50,
  "norm": "slaney",
  "sr":sr,
  "power":2,
  "htk": True,
  "fmin": 0,
  "fmax": sr/2
}

# Extract mel spectrogram
z = librosa.feature.melspectrogram(samples, sr=sr, n_fft=args["n_fft"], hop_length=args["hop_length"], 
                                   win_length=args["win_length"],window=args["window"], 
                                   n_mels=args["n_mels"], center=args["center"],
                                   norm=args["norm"], htk=args["htk"], power=args["power"])
mel_spectrogram = np.log10(z + 1e-6)
mel_spectrogram = np.transpose(mel_spectrogram)
mel_spectrogram2 = mel_spectrogram[np.newaxis,np.newaxis,:, :]

# Pass spectrogram to network
z = model(torch.FloatTensor(mel_spectrogram2))
_, yhat = torch.max(z.data, 1)

CLASSES = 'unknown, yes, no, up, down, left, right, on, off, stop, go'.split(', ')
print("Recognized command: " + CLASSES[yhat])

z = z.cpu().detach().numpy()