import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
import librosa
import sounddevice as sd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchaudio
import os
import concurrent.futures





class AttentionLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ImprovedWakeWordModel(nn.Module):
    def __init__(self, n_classes, sample_rate=22050):
        super().__init__()
        
        self.sample_rate = sample_rate
        window_ms = 25
        self.n_fft = int(sample_rate * window_ms / 1000)
        self.hop_length = self.n_fft // 2
        self.n_mels = 40
        
        # Preprocessor layers
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            n_mels=self.n_mels,
            f_min=100,
            f_max=sample_rate // 2,
            window_fn=torch.hann_window,
            normalized=True
        )
        
        # Заменяем TimeStretch на FrequencyMasking и TimeMasking
        self.spec_augment = nn.Sequential(
            T.FrequencyMasking(freq_mask_param=15),
            T.TimeMasking(time_mask_param=20)
        )
        
        self.amplitude_to_db = T.AmplitudeToDB(
            stype='power',
            top_db=80
        )
        
        # CNN layers with residual connections
        self.layer1 = ResidualBlock(1, 32)
        self.layer2 = ResidualBlock(32, 64)
        self.layer3 = ResidualBlock(64, 128)
        
        # Attention layer
        self.attention = AttentionLayer(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, n_classes)
        
        # Additional layers
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, x, input_is_raw=True, sr=None):
        # Предобработка если нужна
        if input_is_raw:
            x = self.preprocess_audio(x, sr)
            
        # Добавляем размерность канала если её нет
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Применяем аугментацию во время обучения
        if self.training:
            x = self.spec_augment(x)
        
        # Residual blocks with pooling
        x = self.pool(self.layer1(torch.squeeze(x, 1)))
        x = self.pool(self.layer2(x))
        x = self.pool(self.layer3(x))
        
        # Attention mechanism
        x = self.attention(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected layers
        x = self.fc1(x)
        x = self.layer_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

    def preprocess_audio(self, audio_input, sr=None):
        if isinstance(audio_input, (str, Path)):
            waveform, sr = torchaudio.load(audio_input)
        else:
            waveform = audio_input
            sr = sr if sr else self.sample_rate
        
        if isinstance(audio_input, (str, Path)):
            waveform, sr = torchaudio.load(audio_input)
        else:
            waveform = audio_input
            sr = sr if sr else self.sample_rate
                
        # Handle batch dimension
        if waveform.dim() == 3:  # [batch, channels, time]
            # Process each item in batch
            batch_size = waveform.shape[0]
            processed = []
            for i in range(batch_size):
                single_waveform = waveform[i:i+1]  # Keep dimension
                # Process channels
                if single_waveform.shape[1] > 1:
                    single_waveform = torch.mean(single_waveform, dim=1, keepdim=True)
                
                # Normalization
                single_waveform = single_waveform - single_waveform.mean()
                single_waveform = single_waveform / (single_waveform.std() + 1e-8)
                
                # Get mel spectrogram
                mel_spec = self.mel_spec(single_waveform)
                mel_spec_db = self.amplitude_to_db(mel_spec)
                mel_spec_db = (mel_spec_db + 80) / 80
                
                processed.append(mel_spec_db)
                
            return torch.stack(processed)
        else:
        
            
            # Проверяем количество каналов
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Ресемплируем если частота дискретизации отличается
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
                
            # Нормализация
            waveform = waveform - waveform.mean()
            waveform = waveform / (waveform.std() + 1e-8)
            
            # Обрезаем или дополняем до 2 секунд
            target_length = self.sample_rate * 2
            current_length = waveform.shape[1]

            if current_length > target_length:
                waveform = waveform[:, :target_length]
            elif current_length < target_length:
                padding = target_length - current_length
                waveform = F.pad(waveform, (0, padding))

            # Получаем мел-спектрограмму
            mel_spec = self.mel_spec(waveform)
            mel_spec_db = self.amplitude_to_db(mel_spec)
            
            # Нормализация спектрограммы
            mel_spec_db = (mel_spec_db + 80) / 80
            
            return mel_spec_db



def check_audio_signal(waveform):
    """
    Проверка входного аудиосигнала
    """
    # Проверка на наличие нулей
    if torch.all(waveform == 0):
        raise ValueError("Input audio contains only zeros")
        
    # Проверка на наличие NaN или Inf
    if torch.isnan(waveform).any() or torch.isinf(waveform).any():
        raise ValueError("Input audio contains NaN or Inf values")
        
    # Проверка амплитуды
    if waveform.abs().max() < 1e-6:
        print("Warning: Very low amplitude in input audio")



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
        predicted_class = torch.argmax(F.softmax(output), dim=1)
        confidence = torch.max(F.softmax(output), dim=1)[0]
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


def record_audio(duration=2, sample_rate=22050):
    """Record audio from microphone."""
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return audio.flatten()

def predict_from_microphone(audio, model, class_names, device='cuda'):
    """Predict word from microphone input."""
    # Convert to tensor
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0)
    
    # Make prediction
    predicted_class, confidence = predict(model, audio_tensor, sr=22050, device=device)
    
    return class_names[predicted_class], confidence

def real_time_prediction(model, class_names, threshold=0.8, device='cuda'):
    """Continuously predict from microphone with confidence threshold."""
    print("Starting real-time prediction. Press Ctrl+C to stop.")
    
    # Use a ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        try:
            while True:
                # Submit the record_audio function to the executor
                future_audio = executor.submit(record_audio)
                
                # Wait for the recording to complete and get the audio
                audio = future_audio.result()
                
                # Submit the prediction function with the recorded audio
                future_prediction = executor.submit(predict_from_microphone, audio, model, class_names, device)
                
                # Wait for the prediction to complete
                word, confidence = future_prediction.result()
                
                # Check confidence and print prediction
                if confidence > threshold:
                    print(f"Predicted: {word} (confidence: {confidence:.2f})")
                else:
                    print("Confidence too low, no prediction made")

                # Small delay to avoid excessive looping
                # sleep(0.1)

        except KeyboardInterrupt:
            print("\nStopping prediction...")


def train_model(model, train_loader, val_loader, epochs, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-3,
        epochs=epochs,
        steps_per_epoch=len(train_loader)
    )
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        print()
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            # print(output)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        val_loss /= len(val_loader)
        accuracy = (correct / len(val_loader.dataset))*100
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            
        print(f'Epoch: {epoch+1}/{epochs}')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.2f} %')



def collate_fn(batch):
    """
    Функция для обработки батча с разной длиной
    """
    # Разделяем аудио и метки
    waveforms, labels = zip(*batch)
    
    # Находим максимальную длину в батче
    max_length = max(waveform.shape[1] for waveform in waveforms)
    
    # Создаем тензор нужного размера, заполненный нулями
    padded_waveforms = torch.zeros(len(waveforms), 1, max_length)
    
    # Заполняем тензор данными
    for i, waveform in enumerate(waveforms):
        length = waveform.shape[1]
        padded_waveforms[i, :, :length] = waveform
    
    # Преобразуем метки в тензор
    labels = torch.tensor(labels)
    
    return padded_waveforms, labels

class AudioDataset(Dataset):
    def __init__(self, files_and_classes, class_names, transform=None, sample_rate=22050, target_length=None):
        self.files_and_classes = files_and_classes
        self.class_names = class_names
        self.transform = transform
        self.sample_rate = sample_rate
        self.target_length = target_length  # Целевая длина в секундах

    def __len__(self):
        return len(self.files_and_classes)

    def __getitem__(self, idx):
        audio_path, class_name = self.files_and_classes[idx]
        
        # Загрузка аудио
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Приведение к моно если стерео
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Ресемплинг если нужно
        if sample_rate != self.sample_rate:
            resampler = T.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        # Обрезка или дополнение до целевой длины если задана
        if self.target_length is not None:
            target_length = int(self.target_length * self.sample_rate)
            current_length = waveform.shape[1]
            
            if current_length > target_length:
                # Случайно выбираем отрезок нужной длины
                start = random.randint(0, current_length - target_length)
                waveform = waveform[:, start:start + target_length]
            elif current_length < target_length:
                # Дополняем нулями
                padding = target_length - current_length
                waveform = F.pad(waveform, (0, padding))
        
        # Применение трансформаций если есть
        if self.transform:
            waveform = self.transform(waveform)
            
        # Получение метки класса
        label = self.class_names.index(class_name)
        
        return waveform, label

def prepare_data_loaders(data_dir, class_names, batch_size=32, test_size=0.2, val_size=0.1, target_length=2):
    """
    Подготовка train, validation и test даталоадеров
    """
    # Получаем список файлов и классов
    files_and_classes = get_files_and_classes(data_dir)
    
    # Разделяем на train и temp (будущие test + validation)
    train_files, temp_files = train_test_split(
        files_and_classes, 
        test_size=test_size + val_size,
        stratify=[class_name for _, class_name in files_and_classes],
        random_state=42
    )
    
    # Разделяем temp на validation и test
    val_size_adjusted = val_size / (test_size + val_size)
    val_files, test_files = train_test_split(
        temp_files,
        test_size=0.5,
        stratify=[class_name for _, class_name in temp_files],
        random_state=42
    )
    
    # Создаем наборы данных
    train_dataset = AudioDataset(
        train_files, 
        class_names, 
        transform=AudioTransforms(),
        target_length=target_length
    )
    
    val_dataset = AudioDataset(
        val_files, 
        class_names,
        target_length=target_length
    )
    
    test_dataset = AudioDataset(
        test_files, 
        class_names,
        target_length=target_length
    )
    
    # Создаем даталоадеры
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader

class AudioTransforms:
    """
    Класс для аугментации аудиоданных
    """
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, waveform):
        # Добавление случайного шума
        if random.random() < self.p:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
        
        # Случайное изменение громкости
        if random.random() < self.p:
            volume_factor = random.uniform(0.8, 1.2)
            waveform = waveform * volume_factor
            
        # Случайное смещение по времени
        if random.random() < self.p:
            shift = random.randint(0, waveform.shape[1] // 10)
            waveform = torch.roll(waveform, shifts=shift, dims=1)
        
        return waveform
    
    
    
    



def main():
    # Параметры
    data_dir = "audio_data"
    class_names = ["дальше", "следующее", "предыдущее", "пауза", "стоп", "громче", "тише", "шум"]
    batch_size = 32
    epochs = 10
    target_length = 2  # целевая длина в секундах
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Подготовка данных
    train_loader, val_loader, test_loader = prepare_data_loaders(
        data_dir,
        class_names,
        batch_size=batch_size,
        target_length=target_length
    )
    
    # Создание модели
    model = ImprovedWakeWordModel(n_classes=len(class_names)).to(device)
    
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Обучение модели
    train_model(model, train_loader, val_loader, epochs, device)
    
    
    
    # Оценка на тестовом наборе
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for inputs, labels in test_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         outputs = model(inputs)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    
    # print(f'Accuracy on test set: {100 * correct / total}%')
    
    # Сохранение модели
    torch.save(model.state_dict(), 'wake_word_model.pth')
    print("Start")
    real_time_prediction(model, class_names, threshold=0.8, device='cuda')

if __name__ == "__main__":
    main()