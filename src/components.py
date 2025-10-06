import numpy as np
from parameters import SignalParameters


class PRNGenerator:
    """Генерирует PRN-последовательности для L1OCd и L1OCp."""

    
    _GENERATOR_CONFIG = {
        'data': {'N': 1023, 'taps1': [7, 10], 'taps2': [3, 7, 9, 10], 
                 'is1': "0011001000", 'is2_bits': 10},
        'pilot': {'N': 4092, 'taps1': [6, 8, 11, 12], 'taps2': [1, 6],
                  'is1': "000011000101", 'is2_bits': 6}
    }

    def __init__(self, svn: int, code_type: str):
        if code_type not in self._GENERATOR_CONFIG:
            raise ValueError("Тип кода должен быть 'data' или 'pilot'")
        
        self.config = self._GENERATOR_CONFIG[code_type]
        self.svn = svn

    def _lfsr(self, initial_state_bits: list, taps: list, n_bits: int) -> np.ndarray:
        """генерация псевдостроки"""
        state = np.array(initial_state_bits, dtype=np.uint8).copy()
        L = len(state)
        out = np.empty(n_bits, dtype=np.uint8)
        for i in range(n_bits):
            out[i] = state[-1]
            fb = 0
            for tp in taps:
                fb ^= state[tp - 1]
            state[1:] = state[:-1]
            state[0] = fb
        return out


    def generate_sequence(self) -> np.ndarray:
        """Генерирует один полный период PRN-последовательности."""
        is1 = [int(c) for c in self.config['is1']]
        is2_str = format(self.svn, f'0{self.config["is2_bits"]}b')
        is2 = [int(c) for c in is2_str]
        
        g1 = self._lfsr(is1, self.config['taps1'], self.config['N'])
        g2 = self._lfsr(is2, self.config['taps2'], self.config['N'])
        
        prn_sequence = 1 - 2 * np.bitwise_xor(g1, g2)
        return prn_sequence.astype(np.int8)


def generate_repeating_code(params: SignalParameters, rate: float, samples: int) -> np.ndarray:
    """Генерирует повторяющийся код (данные или оверлей)."""
    samples_per_symbol = int(round(params.sampling_rate / rate))
    symbols_needed = int(np.ceil(samples / samples_per_symbol))
    sequence = np.ones(symbols_needed, dtype=np.int8)
    sequence[1::2] = -1 
    
    return np.repeat(sequence, samples_per_symbol)[:samples]


def check_prs_sequence(generated_sequence, reference_start, reference_end, label=""):
    """
    Проверяет корректность сгенерированной PRN-последовательности
    по первым и последним 32 символам, приведённым в ИКД.

    Поддерживает входы в виде 0/1 или ±1.
    Для ±1 выполняется отображение +1 -> 0, -1 -> 1 (соответствие prn = 1 - 2*bit).
    """
    gen = np.asarray(generated_sequence)
    ref_start = np.asarray(reference_start)
    ref_end = np.asarray(reference_end)

    def to_bits(a):
        a = np.asarray(a)
        if a.size == 0:
            raise ValueError("Входной массив пуст.")
        if np.all(np.isin(a, [-1, 1])):
            # -1 -> 1, +1 -> 0
            return (a < 0).astype(int)
        if np.all(np.isin(a, [0, 1])):
            return a.astype(int)
        return (a > 0).astype(int)

    gen_bits = to_bits(gen)
    ref_start_bits = to_bits(ref_start)
    ref_end_bits = to_bits(ref_end)

    if ref_start_bits.size != 32 or ref_end_bits.size != 32:
        raise ValueError("reference_start и reference_end должны содержать по 32 элемента.")

    if gen_bits.size < 32:
        raise ValueError("generated_sequence короче 32 элементов; нечего сравнивать.")

    start_ok = np.array_equal(gen_bits[:32], ref_start_bits)
    end_ok = np.array_equal(gen_bits[-32:], ref_end_bits)

    if start_ok and end_ok:
        print(f"Проверка пройдена: первые и последние 32 символа {label} совпадают с ИКД.")
        return True
    else:
        print(f"Ошибка в проверке PRN последовательности для {label}:")
        if not start_ok:
            print("  – первые 32 символа не совпадают.")
            print(f"    Ожидалось: {ref_start_bits}")
            print(f"    Получено : {gen_bits[:32]}")
        if not end_ok:
            print("  – последние 32 символа не совпадают.")
            print(f"    Ожидалось: {ref_end_bits}")
            print(f"    Получено : {gen_bits[-32:]}")
        return False