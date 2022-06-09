# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import scipy.signal
import tensorflow as tf

from utils import util


def add_delta_deltas(filterbanks, name=None):
    """Compute time first and second-order derivative channels.
      Args:
        filterbanks: float32 tensor with shape [batch_size, len, num_bins, 1]
        name: scope name
      Returns:
        float32 tensor with shape [batch_size, len, num_bins, 3]
    """
    delta_filter = np.array([2, 1, 0, -1, -2])
    delta_delta_filter = scipy.signal.convolve(delta_filter, delta_filter, "full")

    delta_filter_stack = np.array(
        [[0] * 4 + [1] + [0] * 4, [0] * 2 + list(delta_filter) + [0] * 2,
         list(delta_delta_filter)],
        dtype=np.float32).T[:, None, None, :]

    delta_filter_stack /= np.sqrt(
        np.sum(delta_filter_stack ** 2, axis=0, keepdims=True))

    filterbanks = tf.nn.conv2d(
        filterbanks, tf.cast(tf.constant(delta_filter_stack), tf.float32), [1, 1, 1, 1], "SAME", data_format="NHWC",
        name=name)
    return filterbanks


def compute_mel_filterbank_features(
        waveforms,
        sample_rate=16000, dither=1.0 / np.iinfo(np.int16).max, preemphasis=0.97,
        frame_length=25, frame_step=10, fft_length=None,
        window_fn=functools.partial(tf.signal.hann_window, periodic=True),
        lower_edge_hertz=80.0, upper_edge_hertz=7600.0, num_mel_bins=80,
        log_noise_floor=1e-3, apply_mask=True):
    """implement mel-filterbank extraction using tf ops.
      args:
        waveforms: float32 tensor with shape [batch_size, max_len]
        sample_rate: sampling rate of the waveform
        dither: stddev of gaussian noise added to waveform to prevent quantization
          artefacts
        preemphasis: waveform high-pass filtering constant
        frame_length: frame length in ms
        frame_step: frame_step in ms
        fft_length: number of fft bins
        window_fn: windowing function
        lower_edge_hertz: lowest frequency of the filterbank
        upper_edge_hertz: highest frequency of the filterbank
        num_mel_bins: filterbank size
        log_noise_floor: clip small values to prevent numeric overflow in log
        apply_mask: when working on a batch of samples, set padding frames to zero
      returns:
        filterbanks: a float32 tensor with shape [batch_size, len, num_bins, 1]
        masks: masks to indicate padded positions [batch_size, len]
    """
    # `stfts` is a complex64 tensor representing the short-time fourier
    # transform of each signal in `signals`. its shape is
    # [batch_size, ?, fft_unique_bins]
    # where fft_unique_bins = fft_length // 2 + 1

    # find the wave length: the largest index for which the value is !=0
    # note that waveforms samples that are exactly 0.0 are quite common, so
    # simply doing sum(waveforms != 0, axis=-1) will not work correctly.
    # [batch_size]: padding is ok to indicate meaningless points
    wav_lens = tf.reduce_max(
        tf.expand_dims(tf.range(tf.shape(waveforms)[1]), 0) *
        tf.to_int32(tf.not_equal(waveforms, 0.0)),
        axis=-1) + 1
    # adding small noise to the speech for robust modeling
    if dither > 0:
        waveforms += tf.random_normal(tf.shape(waveforms), stddev=dither)
    # time difference, a normal operation to pre-process speech
    if preemphasis > 0:
        waveforms = waveforms[:, 1:] - preemphasis * waveforms[:, :-1]
        wav_lens -= 1
    # frame_length: number of samples in one frame
    frame_length = int(frame_length * sample_rate / 1e3)
    # frame_step: step size => number of frames = sample number // frame_step
    frame_step = int(frame_step * sample_rate / 1e3)
    if fft_length is None:
        fft_length = int(2 ** (np.ceil(np.log2(frame_length))))

    # convert a sequence of audio signals into [num_frames, frame_length]
    # and then apply sfft operation
    # [batch_size, num_frames, fft_unique_bins]
    stfts = tf.signal.stft(
        waveforms,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        window_fn=window_fn,
        pad_end=True)
    # [batch_size, num_frames, frame_length]
    frames = tf.signal.frame(
        waveforms,
        frame_length=frame_length,
        frame_step=frame_step,
        pad_end=True)

    # num_frames: [batch_size] for each sample
    stft_lens = (wav_lens + (frame_step - 1)) // frame_step
    # [batch_size, num_frames]: 1 => valid, 0 => invalid
    masks = tf.to_float(tf.less_equal(
        tf.expand_dims(tf.range(tf.shape(stfts)[1]), 0),
        tf.expand_dims(stft_lens, 1)))

    # an energy spectrogram is the magnitude of the complex-valued stft.
    # a float32 tensor of shape [batch_size, ?, 257].
    magnitude_spectrograms = tf.abs(stfts)

    # warp the linear-scale, magnitude spectrograms into the mel-scale.
    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
    linear_to_mel_weight_matrix = (
        tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
            upper_edge_hertz))
    mel_spectrograms = tf.tensordot(
        magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
    # note: shape inference for tensordot does not currently handle this case.
    mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    log_mel_sgram = tf.log(tf.maximum(log_noise_floor, mel_spectrograms))

    if apply_mask:
        log_mel_sgram *= tf.expand_dims(masks, -1)
        frames *= tf.expand_dims(masks, -1)

    return tf.expand_dims(log_mel_sgram, -1, name="mel_sgrams"), masks, frames


def extract_logmel_features(wav, hparams):
    """ extract logmel features from raw wav file
    
    args:
        wav: [batch, wavlength],
        hparams: hyper-parameters
    returns:
        features: [batch, num_frames, features]
        mask: [batch, num_frames]
    """
    p = hparams
    d = p.audio_num_mel_bins
    mel_fbanks, masks, frames = compute_mel_filterbank_features(
        wav,
        sample_rate=p.audio_sample_rate,
        dither=p.audio_dither,
        preemphasis=p.audio_preemphasis,
        frame_length=p.audio_frame_length,
        frame_step=p.audio_frame_step,
        lower_edge_hertz=p.audio_lower_edge_hertz,
        upper_edge_hertz=p.audio_upper_edge_hertz,
        num_mel_bins=p.audio_num_mel_bins,
        apply_mask=True)
    if p.audio_add_delta_deltas:
        d *= 3
        mel_fbanks = add_delta_deltas(mel_fbanks)

    mfshp = util.shape_list(mel_fbanks)
    mel_fbanks = tf.reshape(mel_fbanks, [mfshp[0], mfshp[1], d])
    masking = tf.expand_dims(masks, -1)

    # this replaces cmvn estimation on data
    var_epsilon = 1e-08
    mean = tf.reduce_sum(mel_fbanks * masking, keepdims=True, axis=1) / \
            (tf.reduce_sum(masking, keepdims=True, axis=1) + var_epsilon)
    sqr_diff = tf.squared_difference(mel_fbanks, mean)
    variance = tf.reduce_sum(sqr_diff * masking, keepdims=True, axis=1) / \
                (tf.reduce_sum(masking, keepdims=True, axis=1) + var_epsilon)

    mel_fbanks = (mel_fbanks - mean) * tf.rsqrt(variance + var_epsilon)

    return mel_fbanks, masks, frames

