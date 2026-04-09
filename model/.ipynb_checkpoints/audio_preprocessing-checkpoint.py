import torch
import torchaudio.transforms as T
import torchaudio.functional as F

TARGET = 16000
N_FFT = 400       # 25ms window at 16kHz
HOP_LENGTH = 160  # 10ms hop
N_MELS = 40


def compute_stft(waveform: torch.Tensor, n_fft: int = N_FFT) -> torch.Tensor:
    """Compute complex STFT spectrogram.

    Args:
        waveform: Waveform tensor. Shape (n_samples,) or (1, n_samples).
        n_fft: FFT size (default: 400).

    Returns:
        Complex STFT of shape (n_freqs, n_frames).
    """
    if waveform.ndim == 2:
        waveform = waveform.squeeze(0)

    window = torch.hann_window(n_fft, device=waveform.device)

    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=HOP_LENGTH,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    return stft  # (n_freqs, n_frames)


def compute_power_spectrogram(waveform: torch.Tensor, n_fft: int = N_FFT) -> torch.Tensor:
    """Compute power spectrogram.

    Args:
        waveform: Waveform tensor. Shape (n_samples,) or (1, n_samples).
        n_fft: FFT size (default: 400).

    Returns:
        Power spectrogram of shape (n_freqs, n_frames).
    """
    stft = compute_stft(waveform, n_fft=n_fft)
    return torch.abs(stft) ** 2  # (n_freqs, n_frames)


def estimate_noise(power_spec: torch.Tensor, n_frames: int = 5) -> torch.Tensor:
    """Estimate noise power by averaging the first n_frames.

    Args:
        power_spec: Power spectrogram of shape (n_freqs, n_frames_total).
        n_frames: Number of initial frames to average.

    Returns:
        Estimated noise profile of shape (n_freqs,).
    """
    return power_spec[:, :n_frames].mean(dim=1)  # (n_freqs,)


def spectral_subtraction(
    power_spec: torch.Tensor,
    noise_profile: torch.Tensor,
    omega: float = 1.5,
    delta: float = 0.5,
) -> torch.Tensor:
    """Perform spectral subtraction.

    Args:
        power_spec: Power spectrogram of shape (n_freqs, n_frames).
        noise_profile: Noise power profile of shape (n_freqs,).
        omega: Scaling factor for noise estimate.
        delta: Floor as a fraction of the original power.

    Returns:
        Denoised power spectrogram of shape (n_freqs, n_frames).
    """
    subtracted = power_spec - omega * noise_profile.unsqueeze(1)
    floor = delta * power_spec
    return torch.max(subtracted, floor)  # (n_freqs, n_frames)


def apply_spectral_gating(
    waveform: torch.Tensor,
    omega: float = 1.5,
    delta: float = 0.02,
    noise_frame: int = 5,
    n_fft: int = N_FFT,
) -> torch.Tensor:
    """Apply spectral gating to reduce noise.

    Computes the power spectrogram, estimates noise from the first few frames,
    and applies spectral subtraction. Returns the denoised power spectrogram
    directly (suitable for passing to power_to_log_mel).

    Note: The notebook version attempted torch.istft on a real-valued tensor,
    which fails. This implementation correctly returns the denoised power
    spectrogram instead.

    Args:
        waveform: Waveform tensor of shape (n_samples,).
        omega: Noise scaling factor.
        delta: Spectral floor fraction.
        noise_frame: Number of initial frames used for noise estimation.

    Returns:
        Denoised power spectrogram of shape (n_freqs, n_frames).
    """
    power_spec = compute_power_spectrogram(waveform, n_fft=n_fft)
    noise_profile = estimate_noise(power_spec, noise_frame)
    return spectral_subtraction(power_spec, noise_profile, omega, delta)


def apply_bandpass(
    waveform: torch.Tensor,
    low_freq: float = 300.0,
    high_freq: float = 3400.0,
    sample_rate: int = TARGET,
) -> torch.Tensor:
    """Apply bandpass filter to waveform.

    Args:
        waveform: Input waveform of shape (n_samples,).
        low_freq: Low cutoff frequency in Hz.
        high_freq: High cutoff frequency in Hz.
        sample_rate: Sample rate in Hz.

    Returns:
        Filtered waveform of shape (n_samples,).
    """
    filtered = F.highpass_biquad(waveform, sample_rate, low_freq)
    filtered = F.lowpass_biquad(filtered, sample_rate, high_freq)
    return filtered


def power_to_log_mel(power_spec: torch.Tensor, n_mels: int = N_MELS, n_fft: int = N_FFT) -> torch.Tensor:
    """Convert power spectrogram to log-mel spectrogram.

    Args:
        power_spec: Power spectrogram of shape (n_freqs, n_frames).
        n_mels: Number of mel filter banks (default: 40).
        n_fft: FFT size used to produce power_spec (default: 400).

    Returns:
        Log-mel spectrogram of shape (n_mels, n_frames).
    """
    mel_scale = T.MelScale(n_mels=n_mels, sample_rate=TARGET, n_stft=n_fft // 2 + 1)

    if power_spec.ndim == 2:
        power_spec = power_spec.unsqueeze(0)  # (1, n_freqs, n_frames)

    mel = mel_scale(power_spec)          # (1, n_mels, n_frames)
    log_mel = T.AmplitudeToDB()(mel)     # (1, n_mels, n_frames)
    return log_mel.squeeze(0)            # (n_mels, n_frames)


def get_log_mel_spectrogram(
    waveform: torch.Tensor,
    filter_method: str = "none",
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    **kwargs,
) -> torch.Tensor:
    """Convert waveform to log-mel spectrogram with optional preprocessing.

    Args:
        waveform: Input waveform of shape (n_samples,).
        filter_method: One of "none", "bandpass", or "spectral_gating".
        n_mels: Number of mel filter banks (default: 40).
        n_fft: FFT size (default: 400). Use 1024 for 128 mel bins to avoid empty filterbanks.
        **kwargs: Passed through to the chosen filter function.

    Returns:
        Log-mel spectrogram of shape (1, n_mels, n_frames).
    """
    if filter_method == "none":
        power = compute_power_spectrogram(waveform, n_fft=n_fft)
    elif filter_method == "bandpass":
        filtered = apply_bandpass(
            waveform,
            low_freq=kwargs.get("low_freq", 300.0),
            high_freq=kwargs.get("high_freq", 3400.0),
            sample_rate=kwargs.get("sample_rate", TARGET),
        )
        power = compute_power_spectrogram(filtered, n_fft=n_fft)
    elif filter_method == "spectral_gating":
        power = apply_spectral_gating(
            waveform,
            omega=kwargs.get("omega", 1.5),
            delta=kwargs.get("delta", 0.02),
            noise_frame=kwargs.get("noise_frame", 5),
            n_fft=n_fft,
        )
    else:
        raise ValueError(
            f"Invalid filter_method '{filter_method}'. "
            "Choose from 'none', 'bandpass', 'spectral_gating'."
        )

    log_mel = power_to_log_mel(power, n_mels=n_mels, n_fft=n_fft)
    return log_mel.unsqueeze(0)  # (1, n_mels, n_frames)
