import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from numpy.fft import fft, ifft, fftshift

from parameters import SignalParameters
from signal_1 import L1OCSignal


def analyze_and_plot(signal: np.ndarray, params: SignalParameters):
    """Выполняет анализ (спектр, АКФ) и строит графики."""
    print("Анализ сигнала...")
    
    # --- Расчет энергетического спектра (PSD) ---
    # Перенос в комплексный базис для анализа
    t = np.arange(len(signal)) / params.sampling_rate
    signal_bb = signal * np.exp(-1j * 2 * np.pi * params.intermediate_freq * t)
    
    freqs, psd = welch(signal_bb, fs=params.sampling_rate, nperseg=params.welch_nperseg, return_onesided=False)
    psd_db = 10 * np.log10(psd / np.max(psd))
    
    # --- Расчет автокорреляционной функции (ACF) ---
    N = len(signal)
    M = 1 << (N - 1).bit_length()
    sig_fft = fft(signal, n=M)
    acf = ifft(sig_fft * np.conj(sig_fft)).real
    
    # Нормализация и центрирование
    autocorr_lin = np.concatenate((acf[-(N - 1):], acf[:N]))
    autocorr_normalized = autocorr_lin / acf[0]
    lags_ms = np.arange(-N + 1, N) / params.sampling_rate * 1000

    # --- Построение графиков ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(fftshift(freqs) / 1e6, fftshift(psd_db))
    ax1.set_title('Энергетический спектр L1OC (Welch)')
    ax1.set_xlabel('Частота, МГц')
    ax1.set_ylabel('Уровень, дБ (норм.)')
    ax1.grid(True)
    
    # Отображаем только центральную часть АКФ
    center_idx = len(lags_ms) // 2
    display_range_ms = 2 
    display_range_samples = int(display_range_ms / 1000 * params.sampling_rate)
    
    start = center_idx - display_range_samples
    end = center_idx + display_range_samples
    ax2.plot(lags_ms[start:end], autocorr_normalized[start:end])
    ax2.set_title('Автокорреляционная функция (центр)')
    ax2.set_xlabel('Лаг, мс')
    ax2.set_ylabel('Норм. значение')
    ax2.grid(True)
    
    plt.tight_layout()
    

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "L1OC_Signal_Analysis.png")
    plt.savefig(output_filename)
    print(f"Графики сохранены в файл: {output_filename}")


if __name__ == '__main__':
    signal_params = SignalParameters(svn=14, duration=0.02)
    l1oc_signal_generator = L1OCSignal(signal_params)
    print("Генерация сигнала...")
    final_signal = l1oc_signal_generator.generate()
    print(f"Сигнал сгенерирован. Количество отсчетов: {len(final_signal)}")
    
    if final_signal is not None:
        analyze_and_plot(final_signal, signal_params)