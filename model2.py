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
            T.FrequencyMasking(freq_mask_param=15).to("cuda"),
            T.TimeMasking(time_mask_param=20).to("cuda")
        )
        
        self.amplitude_to_db = T.AmplitudeToDB(
            stype='power',
            top_db=80
        )
        
        # CNN layers with residual connections
        self.layer1 = ResidualBlock(1, 32)
        self.layer2 = ResidualBlock(32, 32)
        self.layer3 = ResidualBlock(32, 64)
        
        # Attention layer
        self.attention = AttentionLayer(64)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.LazyLinear(512)
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
        output = model(audio_input)
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
    """Record audio from microphone"""
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return audio.flatten()

def predict_from_microphone(model, class_names, device='cuda'):
    """Predict word from microphone input"""
    # Record audio
    audio = record_audio()
    
    # Convert to tensor
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0)
    
    # Make prediction
    predicted_class, confidence = predict(model, audio_tensor, sr=22050, device=device)
    
    return class_names[predicted_class], confidence

def real_time_prediction(model, class_names, threshold=0.8, device='cuda'):
    """Continuously predict from microphone with confidence threshold"""
    print("Starting real-time prediction. Press Ctrl+C to stop.")
    try:
        while True:
            word, confidence = predict_from_microphone(model, class_names, device)
            if confidence > threshold:
                print(f"Predicted: {word} (confidence: {confidence:.2f})")
            else:
                print("Confidence too low, no prediction made")
    except KeyboardInterrupt:
        print("\nStopping prediction...")


def train_model(model, train_loader, val_loader, epochs, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-5,
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
        accuracy = correct / len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model2.pth')
            
        print(f'Epoch: {epoch+1}/{epochs}')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.4f}')



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
    def __init__(self, files_and_classes, class_names, transform=None, sample_rate=22050, 
                 target_length=None, augment_factor=10):
        self.files_and_classes = files_and_classes
        self.class_names = class_names
        self.transform = transform
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.augment_factor = augment_factor  # Сколько аугментированных версий создавать
        
        # Расширяем список файлов для аугментации
        if transform is not None:
            augmented_files = []
            for file_info in files_and_classes:
                augmented_files.append(file_info)  # Оригинальный файл
                for _ in range(augment_factor - 1):  # Аугментированные версии
                    augmented_files.append((file_info[0], file_info[1], True))  # True означает применить аугментацию
            self.files_and_classes = augmented_files
            
            
    def __len__(self):
        return len(self.files_and_classes)
    def __getitem__(self, idx):
        if len(self.files_and_classes[idx]) == 3:
            audio_path, class_name, do_augment = self.files_and_classes[idx]
        else:
            audio_path, class_name = self.files_and_classes[idx]
            do_augment = False
        
        # Загрузка аудио
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Приведение к моно если стерео
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Ресемплинг если нужно
        if sample_rate != self.sample_rate:
            resampler = T.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        # Обрезка или дополнение до целевой длины
        if self.target_length is not None:
            target_length = int(self.target_length * self.sample_rate)
            current_length = waveform.shape[1]
            
            if current_length > target_length:
                start = random.randint(0, current_length - target_length)
                waveform = waveform[:, start:start + target_length]
            elif current_length < target_length:
                padding = target_length - current_length
                waveform = F.pad(waveform, (0, padding))
        
        # Применение аугментации
        if self.transform and do_augment:
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
    Расширенный класс для аугментации аудиоданных
    """
    def __init__(self, p=0.5, sample_rate=22050):
        self.p = p
        self.sample_rate = sample_rate
        
    def add_noise(self, waveform, noise_level=0.005):
        """Добавление случайного шума"""
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    def change_volume(self, waveform, min_factor=0.7, max_factor=1.3):
        """Случайное изменение громкости"""
        volume_factor = random.uniform(min_factor, max_factor)
        return waveform * volume_factor
    
    def time_shift(self, waveform, max_shift_pct=0.2):
        """Случайное смещение по времени"""
        shift = random.randint(0, int(waveform.shape[1] * max_shift_pct))
        return torch.roll(waveform, shifts=shift, dims=1)
    
    def add_background_noise(self, waveform, noise_level=0.1):
        """Добавление цветного шума"""
        noise_type = random.choice(['white', 'pink', 'brown'])
        noise_length = waveform.shape[1]
        
        if noise_type == 'white':
            noise = torch.randn_like(waveform)
        elif noise_type == 'pink':
            # Генерация розового шума
            freqs = torch.fft.rfftfreq(noise_length)
            spectrum = 1 / torch.where(freqs == 0, 1, freqs)**0.5
            noise = torch.fft.irfft(spectrum * torch.randn(spectrum.shape[0]))
            noise = noise[:waveform.shape[1]].unsqueeze(0)
        else:  # brown noise
            # Генерация коричневого шума
            freqs = torch.fft.rfftfreq(noise_length)
            spectrum = 1 / torch.where(freqs == 0, 1, freqs)
            noise = torch.fft.irfft(spectrum * torch.randn(spectrum.shape[0]))
            noise = noise[:waveform.shape[1]].unsqueeze(0)
        
        return waveform + noise_level * noise / noise.std()
    
    def pitch_shift(self, waveform, max_steps=3):
        """Изменение высоты тона"""
        steps = random.uniform(-max_steps, max_steps)
        # Используем stretch как аппроксимацию pitch shift
        stretch_factor = 2 ** (steps / 12)
        length = waveform.shape[1]
        new_length = int(length * stretch_factor)
        
        if new_length > length:
            waveform = F.interpolate(
                waveform.unsqueeze(0), 
                size=new_length, 
                mode='linear', 
                align_corners=False
            ).squeeze(0)
            # Обрезаем до исходной длины
            waveform = waveform[:, :length]
        else:
            waveform = F.interpolate(
                waveform.unsqueeze(0), 
                size=new_length, 
                mode='linear', 
                align_corners=False
            ).squeeze(0)
            # Дополняем до исходной длины
            padding = length - new_length
            waveform = F.pad(waveform, (0, padding))
            
        return waveform
    
    def time_stretch(self, waveform, min_speed=0.8, max_speed=1.2):
        """Растяжение/сжатие во времени"""
        speed_factor = random.uniform(min_speed, max_speed)
        new_length = int(waveform.shape[1] / speed_factor)
        
        waveform = F.interpolate(
            waveform.unsqueeze(0),
            size=new_length,
            mode='linear',
            align_corners=False
        ).squeeze(0)
        
        # Приводим к исходной длине
        if waveform.shape[1] > waveform.shape[1]:
            waveform = waveform[:, :waveform.shape[1]]
        else:
            padding = waveform.shape[1] - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
            
        return waveform
    
    def __call__(self, waveform):
        augmented_waveform = waveform.clone()
        
        # Применяем каждое преобразование с вероятностью p
        if random.random() < self.p:
            augmented_waveform = self.add_noise(augmented_waveform)
        
        if random.random() < self.p:
            augmented_waveform = self.change_volume(augmented_waveform)
        
        if random.random() < self.p:
            augmented_waveform = self.time_shift(augmented_waveform)
        
        if random.random() < self.p:
            augmented_waveform = self.add_background_noise(augmented_waveform)
        
        if random.random() < self.p:
            augmented_waveform = self.pitch_shift(augmented_waveform)
        
        if random.random() < self.p:
            augmented_waveform = self.time_stretch(augmented_waveform)
        
        return augmented_waveform
    
    
    
class AudioModelWrapper:
    def __init__(self, model):
        self.model = model.to('cuda')
        
        # Трассировка модели для оптимизации
        self.traced_model = torch.jit.trace(self.model, torch.rand(1, 1, 80, 80).to('cuda'))
        
        # Аугментация и нормализация
        self.volume_normalization = T.Vol(gain=0.5, gain_type="amplitude")
        
    def preprocess(self, waveform):
        # Нормализация громкости
        waveform = self.volume_normalization(waveform)
        
        # Преобразование в мел-спектрограмму
        spectrogram = T.MelSpectrogram()(waveform).log2()
        
        return spectrogram.unsqueeze(0).to('cuda')

    def predict(self, waveform):
        # Используем torch.cuda.amp для ускорения предсказания
        with torch.no_grad(), torch.cuda.amp.autocast():
            input_data = self.preprocess(waveform)
            prediction = self.traced_model(input_data)
        return prediction



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
    
    model.load_state_dict(torch.load('best_model2.pth'))
    
    # Обучение модели
    train_model(model, train_loader, val_loader, epochs, device)
    
    
    
    # Оценка на тестовом наборе
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy on test set: {100 * correct / total}%')
    
    # Сохранение модели
    # torch.save(model.state_dict(), 'wake_word_model.pth')
    
    # model = ImprovedWakeWordModel(n_classes=len(class_names))
    model.load_state_dict(torch.load('best_model2.pth'))
    print("Start")
    real_time_prediction(model, class_names, threshold=0.8, device='cuda')

if __name__ == "__main__":
    main()