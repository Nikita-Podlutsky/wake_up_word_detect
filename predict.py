import time
import torch
from model2 import ImprovedWakeWordModel
from model import predict_from_microphone
import queue
import threading
import numpy as np
import sounddevice as sd

import queue
import threading
import numpy as np
import sounddevice as sd

class AudioProcessor:
    def __init__(self, model, class_names, buffer_duration=2, sample_rate=22050, 
                 threshold=0.8, device='cuda', volume_multiplier=1.5):
        self.model = model
        self.class_names = class_names
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.threshold = threshold
        self.device = device
        self.volume_multiplier = volume_multiplier
        
        self.audio_queue = queue.Queue()
        self.is_running = False
        
        self.buffer_size = int(buffer_duration * sample_rate)
        self.audio_buffer = np.zeros(self.buffer_size)
        self.last_prediction_time = 0

    def adjust_volume(self, audio):
        """Простое усиление громкости"""
        # Умножаем сигнал на коэффициент и ограничиваем значения
        amplified = audio * self.volume_multiplier
        return np.clip(amplified, -1.0, 1.0)  # Предотвращаем клиппинг

    def audio_callback(self, indata, frames, time, status):
        """Callback для аудио потока"""
        if status:
            print(f"Status: {status}")
        self.audio_queue.put(indata.copy())
    
    def process_audio(self):
        """Обработка аудио и получение предсказаний"""
        while self.is_running:
            try:
                new_audio = self.audio_queue.get(timeout=1).flatten()
                
                # Усиливаем громкость
                amplified_audio = self.adjust_volume(new_audio)
                
                # Обновляем буфер
                self.audio_buffer = np.roll(self.audio_buffer, -len(amplified_audio))
                self.audio_buffer[-len(amplified_audio):] = amplified_audio
                
                # Делаем предсказание
                word, confidence = predict_from_microphone(
                    self.audio_buffer,
                    self.model,
                    self.class_names,
                    self.device
                )
                if word != "шум":
                    if confidence > self.threshold:
                        if (time.time() - self.last_prediction_time) > 1: 
                            print(f"Predicted: {word} (confidence: {confidence:.2f})")
                            self.last_prediction_time = time.time()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing: {e}")
    
    def start(self):
        """Запуск непрерывной обработки"""
        self.is_running = True
        
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.start()
        
        with sd.InputStream(callback=self.audio_callback,
                          channels=1,
                          samplerate=self.sample_rate,
                          blocksize=int(self.sample_rate * 0.1)):
            print("Starting real-time prediction. Press Ctrl+C to stop.")
            try:
                while self.is_running:
                    sd.sleep(100)
            except KeyboardInterrupt:
                self.stop()
    
    def stop(self):
        """Остановка обработки"""
        self.is_running = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
        print("\nStopping prediction...")

def real_time_prediction(model, class_names, threshold=0.8, device='cuda', volume_multiplier=2):
    processor = AudioProcessor(
        model, 
        class_names, 
        threshold=threshold, 
        device=device,
        volume_multiplier=volume_multiplier
    )
    processor.start()

class_names = ["дальше", "следующее", "предыдущее", "пауза", "стоп", "громче", "тише", "шум"]
device = "cuda"

model = ImprovedWakeWordModel(n_classes=len(class_names)).to(device)
model.load_state_dict(torch.load('best_model2.pth'))
model.eval()

real_time_prediction(model, class_names, threshold=0.95, device='cuda')