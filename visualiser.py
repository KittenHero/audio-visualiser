import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as fa
import soundfile as sf
import sounddevice as sd
import time
from contextlib import suppress
from itertools import accumulate

def setup():
    with suppress(ValueError):
        sd.query_devices('pulse')
        sd.default.device = 'pulse'

def visualise(audiofile):
    data, subfig = {}, {}
    audio, sample_rate = sf.read(audiofile, dtype='float64')
    nsample, nchannel = audio.shape

    interval = 25
    skip = 32
    n_bands = 32
    buffer_size = 43
    n_history = 1024

    def setup_plot():
        data['energy_buffer'] = np.zeros((nchannel, buffer_size, n_bands))
        data['shift_buffer'] = 0
        time_axis = [- i/sample_rate for i in range(sample_rate)[::-skip]]
        data['audio_sample'] = np.array([[ch]*len(time_axis) for ch in range(nchannel)], dtype=np.float64)

        fig, ax = plt.subplots(3, 1)
        data['fig'] = fig
        subfig['sample'] = ax[0].plot(time_axis, data['audio_sample'].T)
        subfig['freq'] = ax[1].plot([1], [[1, -1]])
        subfig['energy'] = ax[2].plot(range(n_bands), np.zeros((n_bands, nchannel)))

        limits = (time_axis[0], time_axis[-1], audio.min(), nchannel - 1 + audio.max())
        ax[0].axis(limits)
        ax[1].axis((0, 20000, -200, 200))
        ax[2].axis((0, n_bands, 0, 40))

        for axis in ax:
            axis.set_axis_off()
        fig.set_facecolor('black')
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0, wspace=0)

    def update(frame_data):
        offset, shift = frame_data
        updated = update_sample(offset, shift) + update_freq(offset, shift) + update_beat(offset + shift)
        return updated

    def timer():
        begin = ltick = time.time()
        sd.play(audio / 2, sample_rate)
        offset = shift = 0
        while offset < nsample:
            tick = time.time()
            offset = int(sample_rate * (ltick - begin))
            shift = int(sample_rate * (tick - ltick))
            yield offset, shift
            ltick = tick

    def update_sample(offset, shift):
        if offset >= nsample:
            return []

        if shift > sample_rate:
            offset += shift // sample_rate * sample_rate
            shift %= sample_rate
        dshift = (skip - 1 + shift) // skip
        data['audio_sample'] = np.roll(data['audio_sample'], -dshift, axis=1)

        for ch, (line, ch_sample) in enumerate(zip(subfig['sample'], data['audio_sample'])):
            if offset + shift < nsample:
                ch_sample[-dshift:] = ch + audio[offset:offset + shift:skip, ch]
            else:
                ch_sample[-dshift:((nsample + skip - offset - 1) // skip) - dshift] = ch + audio[offset::skip, ch]

            line.set_ydata(ch_sample)

        return subfig['sample']

    def update_freq(offset, shift):
        if offset >= nsample:
            return []

        freq_axis = np.fft.rfftfreq(min(shift, nsample - offset) or 1, 1 / sample_rate)
        for ch, line in enumerate(subfig['freq']):
            if offset + shift < nsample:
                fft = np.abs(np.fft.rfft(audio[offset:offset + shift, ch]))
            else:
                fft = np.abs(np.fft.rfft(audio[offset:, ch]))
            line.set_xdata(freq_axis)
            line.set_ydata((ch%2 * 2 - 1)*fft)
        return subfig['freq']

    def update_beat(offset):
        shift_buffer = data['shift_buffer']
        if offset - shift_buffer < n_history or offset >= nsample:
            return []
        n_update = (offset - shift_buffer) // n_history
        data['shift_buffer'] += n_update*n_history
        data['energy_buffer'] = np.roll(data['energy_buffer'], -n_update, axis=0)
        energy_buffer = data['energy_buffer']
        for ch, (line, ch_buff) in enumerate(zip(subfig['energy'], energy_buffer)):
            for i in range(n_update):
                if n_update - i >= n_history:
                    continue
                energy = np.abs(np.fft.rfft(audio[shift_buffer + i*n_history:shift_buffer + (i + 1)*n_history, ch]))**2
                ch_buff[i - n_update] = np.array([
                    (j + 1)/n_history * np.sum(energy[k:k + j + 1])
                    for j, k in enumerate(accumulate(range(n_bands)))
                ])
            line.set_ydata(ch_buff[-1])
        return subfig['energy']

    setup_plot()
    animation = fa(data['fig'], update, frames=timer, interval=interval, repeat=0, blit=1)
    plt.show()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='idk it does stuff')
    parser.add_argument('file', metavar='f', help='wav file')
    args = parser.parse_args()
    setup()
    visualise(args.file)
