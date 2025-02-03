import os
import soundfile as sf
from TTS.api import TTS
from datasets import load_dataset
import os
import soundfile as sf
import random
import glob


tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")




def audio_generator(text,filename, speaker_wav):
    wav = tts.tts(text, speaker_wav=speaker_wav,
                      language="ru", split_sentences=False)
    sf.write(filename, wav, 22050)


def t2s(text):
    if not os.path.exists(f"audio_data/{text}"):
        os.makedirs(f"audio_data/{text}")
    k = len(glob.glob(f"audio_data/{text}/*"))
    for i in range(4000):
        if i < k:
            continue
        print(i)
        sample = dataset['train'][random.randint(0,19880)]
        audio = sample["audio"]["array"]
        sample_rate = sample["audio"]["sampling_rate"]
        sf.write("speaker_wav.wav", audio, sample_rate)
        audio_generator(text, f"audio_data/{text}/{i}.wav","speaker_wav.wav")
        
# Загружаем датасет
dataset = load_dataset("mozilla-foundation/common_voice_11_0", "ru",trust_remote_code=True)

# Список команд
commands = ["дальше", "следующее", "предыдущее", "пауза", "стоп", "громче", "тише"]
# commands = ["предыдущее", "пауза", "стоп", "громче", "тише"]
for i in commands:
    t2s(i)