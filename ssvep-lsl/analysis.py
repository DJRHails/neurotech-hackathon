"""Canonical Correlation Analysis for SSVEP classification."""

import numpy as np
from sklearn.cross_decomposition import CCA


class CCAAnalysis:
    """CCA-based SSVEP frequency detector.

    Generates synthetic sine/cosine reference signals at each
    target frequency (with harmonics) and computes canonical
    correlation against incoming EEG data.
    """

    def __init__(
        self,
        freqs: list[float],
        win_len: float,
        s_rate: int,
        n_harmonics: int = 2,
    ) -> None:
        self.freqs = freqs
        self.win_len = win_len
        self.s_rate = s_rate
        self.n_harmonics = n_harmonics
        self.references = self._build_references()
        self.cca = CCA(n_components=1)

    def _build_references(self) -> dict[float, np.ndarray]:
        """Build sine/cosine reference signals per frequency."""
        n_samples = int(self.s_rate * self.win_len)
        t = np.linspace(0, self.win_len, n_samples)
        refs: dict[float, np.ndarray] = {}
        for freq in self.freqs:
            components = []
            for h in range(1, self.n_harmonics + 1):
                components.append(np.sin(2 * np.pi * h * freq * t))
                components.append(np.cos(2 * np.pi * h * freq * t))
            refs[freq] = np.array(components).T
        return refs

    def apply_cca(self, eeg: np.ndarray) -> list[float]:
        """Compute CCA scores for each target frequency.

        Args:
            eeg: EEG array (n_samples, n_channels).

        Returns:
            Correlation score per target frequency.
        """
        scores = []
        for freq in self.freqs:
            ref = self.references[freq]
            sig_c, ref_c = self.cca.fit_transform(eeg, ref)
            scores.append(float(np.corrcoef(sig_c.T, ref_c.T)[0, 1]))
        return scores


if __name__ == "__main__":
    test_freqs = [7, 10, 15]
    t_len = 2
    sample_rate = 250
    t_vec = np.linspace(0, t_len, sample_rate * t_len)

    test_sig = np.sin(2 * np.pi * 10 * t_vec) + 0.05 * np.random.rand(len(t_vec))
    cca = CCAAnalysis(freqs=test_freqs, win_len=t_len, s_rate=sample_rate, n_harmonics=2)
    result = cca.apply_cca(test_sig[:, np.newaxis])
    print(result)
