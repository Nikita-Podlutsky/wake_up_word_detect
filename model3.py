import os
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from pathlib import Path

class WakeWordModel(nn.Module):
    def __init__(self, n_classes, sample_rate=16000):
        super(WakeWordModel, self).__init__()
        
        self.sample_rate = sample_rate
        self.n_mels = 128  # Увеличили количество мел-фильтров
        self.n_fft = 512   # Увеличили размер окна FFT
        self.hop_length = 160
        self.win_length = 512
        
        # Определение мел-спектрограммы с улучшенными параметрами
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            f_min=20,
            f_max=8000,
            power=2.0
        )
        
        # Используем аугментацию для улучшения устойчивости модели к различным шумам и изменениям
        self.spec_augment = nn.Sequential(
            T.TimeStretch(),  # Применение растяжки времени
            T.FrequencyMasking(freq_mask_param=30),  # Применение маскировки в частотной области
            T.TimeMasking(time_mask_param=100)  # Применение маскировки во временной области
        )

        # Преобразуем амплитуду в децибелы для улучшенной визуализации спектра
        self.amplitude_to_db = T.AmplitudeToDB(stype='power')
        
        # Улучшенные CNN слои
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Dropout2d(0.3)
        )
        
        # Добавим рекуррентный слой
        self.gru = nn.GRU(input_size=256, hidden_size=128, num_layers=2, 
                         batch_first=True, bidirectional=True, dropout=0.3)
        
        # Полносвязные слои
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )
        
    def preprocess_audio(self, audio_input, sr=None, augment=False):
        if isinstance(audio_input, (str, Path)):
            waveform, sr = torchaudio.load(audio_input)
        else:
            waveform = audio_input
            sr = sr if sr else self.sample_rate
            
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Улучшенная нормализация
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)
        
        target_length = self.sample_rate * 2
        current_length = waveform.shape[1]

        if current_length > target_length:
            start = torch.randint(0, current_length - target_length, (1,))
            waveform = waveform[:, start:start + target_length]
        else:
            waveform = nn.functional.pad(waveform, (0, target_length - current_length))

        # Получаем мел-спектрограмму
        mel_spec = self.mel_spec(waveform)
        
        if augment:
            mel_spec = self.spec_augment(mel_spec)
            
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        # Улучшенная нормализация спектрограммы
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        return mel_spec_db
        
    def forward(self, x, input_is_raw=True, sr=None):
        if input_is_raw:
            x = self.preprocess_audio(x, sr, augment=self.training)
            
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        # CNN feature extraction
        x = self.features(x)
        
        # Подготовка для GRU
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 2, 3, 1)  # [batch, height, width, channels]
        x = x.reshape(batch_size, height, width * channels)
        
        # GRU
        x, _ = self.gru(x)
        x = x[:, -1, :]  # берем последний выход
        
        # Classification
        x = self.classifier(x)
        
        return torch.log_softmax(x, dim=1)
def train_step(model, optimizer, criterion, data, labels, device):
    model.train()
    optimizer.zero_grad()
    
    # Перемещаем данные на устройство
    data, labels = data.to(device), labels.to(device)
    
    # Прямой проход
    outputs = model(data, input_is_raw=True)

    loss = criterion(outputs, labels)
    
    # Обратный проход
    loss.backward()
    optimizer.step()
    
    return loss.item()

def predict(model, audio_input, sr=None, device='cuda'):
    model.eval()
    with torch.no_grad():
        # Перемещаем модель на нужное устройство
        model = model.to(device)
        
        # Если входные данные - тензор, перемещаем их на устройство
        if isinstance(audio_input, torch.Tensor):
            audio_input = audio_input.to(device)
            
        # Получаем предсказание
        output = model(audio_input, input_is_raw=True, sr=sr)
        predicted_class = torch.argmax(output, dim=1)
        confidence = torch.max(output, dim=1)[0]
        
        return predicted_class.cpu().item(), confidence.cpu().item()

def get_files_and_classes(directory):
    files_and_classes = []

    # Обходим все директории и файлы в указанной папке
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.wav'):  # Можно изменить на любой нужный формат
                # Получаем название класса (имя поддиректории)
                class_name = os.path.basename(dirpath)
                # Создаем полный путь к файлу
                full_path = os.path.join(dirpath, filename)
                # Добавляем в список в виде [путь, класс]
                files_and_classes.append([full_path, class_name])
    
    return files_and_classes

# Пример использования:
def main():
    epochs = 100
    class_names = ["дальше", "следующее", "предыдущее", "пауза", "стоп", "громче", "тише"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WakeWordModel(n_classes=len(class_names)).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    for i in range(epochs):
        total_loss = 0
        n_samples = 0
        
        for path, class_name in tqdm(get_files_and_classes("audio_data")[:100]):
            audio_tensor = torchaudio.load(path)[0]
            label_idx = class_names.index(class_name)
            labels = torch.tensor([label_idx])
            
            loss = train_step(model, optimizer, criterion, audio_tensor, labels, device)
            total_loss += loss
            n_samples += 1
        
        avg_loss = total_loss / n_samples
        print(f"Epoch [{i+1}/{epochs}], Average Loss: {avg_loss:.4f}")

    # Добавьте код для сохранения модели
    torch.save(model.state_dict(), 'wake_word_model.pth')
    print(model(torchaudio.load("path")[0]))


if __name__ == "__main__":
    main()
