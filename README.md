
# Wake-Up Word Detection Project

This project is focused on building and deploying a wake-up word detection system using machine learning and audio processing techniques. The goal is to create a model that can recognize a specific word or phrase from audio input in real time.

## Features

- **Audio Processing**: Uses `librosa`, `pydub`, and other libraries for pre-processing and feature extraction from audio.
- **Deep Learning**: Built on top of `Keras` and `PyTorch`, using neural networks to train and deploy the wake-up word detector.
- **Voice Activity Detection**: Integrated with `pyannote-audio` for speaker separation and voice activity detection (VAD).
- **Speech Features**: Uses `python-speech-features` for extracting relevant features from the audio signal for better model performance.
- **Speech Synthesis**: Can generate responses using the `TTS` library.
- **Real-Time Prediction**: Capable of detecting wake-up words in real time using `sounddevice`.

## Installation

To set up the project, you'll need to have `Poetry` installed on your machine for managing dependencies and the virtual environment.

1. **Install Poetry** (if you haven't already):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd wake-up-word
   ```

3. **Install dependencies**:
   ```bash
   poetry install
   ```

4. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

## Dependencies

This project requires Python version 3.11 or higher. Below are the key dependencies included in the project:

- `matplotlib` - For plotting and visualizations.
- `seaborn` - For better data visualization.
- `keras` - For deep learning model implementation.
- `pydub` & `librosa` - For audio processing and feature extraction.
- `scikit-learn` - For additional machine learning utilities.
- `torch` - For deep learning model implementation (PyTorch).
- `pyannote-audio` - For speaker separation and voice activity detection.
- `nltk` - For natural language processing tasks.
- `sounddevice` - For real-time audio input.

## Usage

Once you've installed the dependencies and activated the virtual environment, you can run the wake-up word detection system. For example:

1. **Run the wake-up word detection**:
   ```bash
   python detect_wake_word.py
   ```

2. **Train the model**:
   If you'd like to train the model with your own dataset, run:
   ```bash
   python train_model.py
   ```

3. **Generate speech response**:
   You can use the TTS library to generate speech output:
   ```bash
   python generate_response.py --text "Hello, how can I help you?"
   ```

## Example

Here's an example of using the wake-up word detection model:

```python
import sounddevice as sd
import numpy as np
from wake_up_word_model import detect_wake_word

# Set up the audio input stream
sample_rate = 16000
duration = 5  # Duration of the recording in seconds

# Function to process audio and detect the wake-up word
def process_audio(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_data = np.array(indata)
    detected = detect_wake_word(audio_data)
    if detected:
        print("Wake-up word detected!")

# Record audio in real-time
with sd.InputStream(callback=process_audio, channels=1, samplerate=sample_rate):
    sd.sleep(duration * 1000)
```

## Contributing

If you'd like to contribute to the project, feel free to fork the repository, submit issues, and create pull requests. Here are some ways you can contribute:

- Improve the accuracy of the wake-up word detection model.
- Enhance real-time performance of the system.
- Add more features or audio processing techniques.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

