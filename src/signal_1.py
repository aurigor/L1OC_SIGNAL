import numpy as np
from parameters import SignalParameters
from components import PRNGenerator, generate_repeating_code, check_prs_sequence


class L1OCSignal:
    """Собирает и модулирует сигнал ГЛОНАСС L1OC."""

    def __init__(self, params: SignalParameters):
        self.params = params
        self.num_samples = int(round(params.duration * params.sampling_rate))
        self.t = np.arange(self.num_samples) / params.sampling_rate
        self.if_signal = None

    def _upsample_prn(self, prn_sequence: np.ndarray, prn_rate: float) -> np.ndarray:
        """Повышает дискретизацию PRN-кода до частоты дискретизации."""
        samples_per_chip = int(round(self.params.sampling_rate / prn_rate))
        chips_needed = int(np.ceil(self.params.duration * prn_rate))
       
        prn_chips = np.tile(prn_sequence, int(np.ceil(chips_needed / len(prn_sequence))))[:chips_needed]
        return np.repeat(prn_chips, samples_per_chip)[:self.num_samples]

    def generate(self):
        """Основной метод, генерирующий сигнал на промежуточной частоте."""
        prn_clock_rate = self.params.f_t1 / 2
        prn_d_gen = PRNGenerator(self.params.svn, 'data')
        prn_p_gen = PRNGenerator(self.params.svn, 'pilot')
        prn_d_samples = self._upsample_prn(prn_d_gen.generate_sequence(), prn_clock_rate)
        prn_p_samples = self._upsample_prn(prn_p_gen.generate_sequence(), prn_clock_rate)

        check_prs_sequence(prn_d_gen.generate_sequence(), self.params.table_prn_d_first, self.params.table_prn_d_end, "data")
        check_prs_sequence(prn_p_gen.generate_sequence(), self.params.table_prn_p_first, self.params.table_prn_p_end, "pilot")


        nav_samples = generate_repeating_code(self.params, self.params.nav_data_rate, self.num_samples)
        overlay_samples = generate_repeating_code(self.params, self.params.overlay_rate, self.num_samples)

        data_comp = prn_d_samples * nav_samples * overlay_samples
        
        boc_subcarrier = np.sign(np.sin(2 * np.pi * self.params.f_t1 * self.t))
        pilot_comp = prn_p_samples * boc_subcarrier

        slot_len = int(round(self.params.sampling_rate / self.params.f_t1))
        
        # Создаем маску [1,0,0,0] для DPPP
        num_slots = int(np.ceil(self.num_samples / slot_len))
        num_repeats = int(np.ceil(num_slots / len(self.params.tdm_pattern)))
        slot_source = np.tile(self.params.tdm_pattern, num_repeats)[:num_slots]
        tdm_mask = np.repeat(slot_source, slot_len)[:self.num_samples]

        baseband_q = np.where(tdm_mask == 1, data_comp, pilot_comp)
        baseband_i = np.zeros(self.num_samples)

        carrier_cos = np.cos(2 * np.pi * self.params.intermediate_freq * self.t)
        carrier_sin = np.sin(2 * np.pi * self.params.intermediate_freq * self.t)
        
        self.if_signal = baseband_i * carrier_cos - baseband_q * carrier_sin
        return self.if_signal


