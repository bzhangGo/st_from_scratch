# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml
import numpy as np
import librosa
from utils.util import batch_indexer, token_indexer


def audio_encode(wav_path, offset=0.0, duration=None, sample_rate=16000):
    """
    Encoding audio files into float list given the offset and duration
    We assume the sample rate to be 16k.
    """
    # load data, sr=None enforce to use the native sample rate
    data, rate = librosa.load(wav_path, sr=None, offset=offset, duration=duration)
    if sample_rate is not None and rate != sample_rate:
        data, rate = librosa.load(wav_path, sr=sample_rate, offset=offset, duration=duration)
    assert len(data.shape) == 1 and rate == sample_rate, (data.shape, rate)

    if data.dtype not in [np.float32, np.float64]:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    return data.astype(np.float32)


def get_rough_length(audio_infor, p):
    duration = audio_infor['duration']  # in seconds
    # total signals
    num_signal = int(duration * p.audio_sample_rate)
    # windows properties
    frame_step = int(p.audio_frame_step * p.audio_sample_rate / 1e3)
    # total frame
    num_frame = (num_signal + frame_step - 1) // frame_step
    return num_frame


class Dataset(object):
    def __init__(self, params, src_file, tgt_file,
                 src_vocab, tgt_vocab,
                 batch_or_token='batch',
                 data_leak_ratio=0.5,
                 src_audio_path=''):
        self.source = src_file
        self.target = tgt_file
        self.src_vocab = src_vocab         # Note source vocabulary here is meaningless
        self.tgt_vocab = tgt_vocab
        self.batch_or_token = batch_or_token
        self.data_leak_ratio = data_leak_ratio

        self.p = params
        self.sr = params.audio_sample_rate
        self.src_audio_path = src_audio_path

        self.max_frame_len = params.max_frame_len
        self.max_text_len = params.max_text_len

        self.leak_buffer = []

    # loading dataset
    def load_data(self, is_train=False):
        sources = self.source.strip().split(";")
        targets = self.target.strip().split(";")

        for source, target in zip(sources, targets):
            with open(source, 'r', encoding='utf-8') as src_reader, \
                    open(target, 'r', encoding='utf-8') as tgt_reader:

                while True:
                    src_line = src_reader.readline()
                    tgt_line = tgt_reader.readline()

                    if tgt_line == "" or src_line == "":
                        break

                    src_line = src_line.strip()
                    tgt_line = tgt_line.strip()

                    if is_train and (tgt_line == "" or src_line == ""):
                        continue

                    yield (
                        yaml.safe_load(src_line)[0],
                        self.tgt_vocab.to_id(tgt_line.split()[:self.max_text_len])
                    )

    def to_matrix(self, batch):
        batch_size = len(batch)

        # handle source audios
        sources = []
        frames = []
        for sample in batch:
            audio_infor = sample[1]
            frames.append(get_rough_length(audio_infor, self.p))

            sources.append(audio_encode(
                os.path.join(self.src_audio_path, audio_infor['wav']),
                audio_infor['offset'],
                audio_infor['duration'],
                sample_rate=self.sr))
 
        src_lens = [len(sample) for sample in sources]
        tgt_lens = [len(sample[2]) for sample in batch]

        src_len = min(self.max_frame_len, max(src_lens))
        tgt_len = min(self.max_text_len, max(tgt_lens))

        s = np.zeros([batch_size, src_len], dtype=np.float32)
        t = np.zeros([batch_size, tgt_len], dtype=np.int32)
        x = []
        for eidx, sample in enumerate(batch):
            x.append(sample[0])
            src_ids, tgt_ids = sources[eidx], sample[2]

            s[eidx, :min(src_len, len(src_ids))] = src_ids[:src_len]
            t[eidx, :min(tgt_len, len(tgt_ids))] = tgt_ids[:tgt_len]

        # construct sparse label sequence, for ctc training
        seq_indexes = []
        seq_values = []
        for n, sample in enumerate(batch):
            sequence = sample[2][:tgt_len]

            seq_indexes.extend(zip([n] * len(sequence), range(len(sequence))))
            seq_values.extend(sequence)

        seq_indexes = np.asarray(seq_indexes, dtype=np.int64)
        seq_values = np.asarray(seq_values, dtype=np.int32)
        seq_shape = np.asarray([batch_size, tgt_len], dtype=np.int64)

        return x, s, t, (seq_indexes, seq_values, seq_shape), frames

    def processor(self, batch):
        x, s, t, spar, f = self.to_matrix(batch)
        return {
            'src': s,
            'tgt': t,
            'frames': f,
            'spar': spar,
            'index': x,
            'raw': batch,
        }

    def batcher(self, size, buffer_size=1000, shuffle=True, train=True):
        def _handle_buffer(_buffer):
            sorted_buffer = sorted(
                _buffer, key=lambda xx: max(get_rough_length(xx[1], self.p), len(xx[2])))

            if self.batch_or_token == 'batch':
                buffer_index = batch_indexer(len(sorted_buffer), size)
            else:
                buffer_index = token_indexer(
                    [[get_rough_length(sample[1], self.p), len(sample[2])] for sample in sorted_buffer], size)

            index_over_index = batch_indexer(len(buffer_index), 1)
            if shuffle: np.random.shuffle(index_over_index)

            for ioi in index_over_index:
                index = buffer_index[ioi[0]]
                batch = [sorted_buffer[ii] for ii in index]
                yield batch

        buffer = self.leak_buffer
        self.leak_buffer = []
        for i, (src_ids, tgt_ids) in enumerate(self.load_data(train)):
            buffer.append((i, src_ids, tgt_ids))
            if len(buffer) >= buffer_size:
                for data in _handle_buffer(buffer):
                    # check whether the data is tailed
                    batch_size = len(data) if self.batch_or_token == 'batch' \
                        else max(sum([len(sample[2]) for sample in data]), sum([get_rough_length(sample[1], self.p) for sample in data]))
                    if batch_size < size * self.data_leak_ratio:
                        self.leak_buffer += data
                    else:
                        yield data
                buffer = self.leak_buffer
                self.leak_buffer = []

        # deal with data in the buffer
        if len(buffer) > 0:
            for data in _handle_buffer(buffer):
                # check whether the data is tailed
                batch_size = len(data) if self.batch_or_token == 'batch' \
                    else max(sum([len(sample[2]) for sample in data]), sum([get_rough_length(sample[1], self.p) for sample in data]))
                if train and batch_size < size * self.data_leak_ratio:
                    self.leak_buffer += data
                else:
                    yield data
