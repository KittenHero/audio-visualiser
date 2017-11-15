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
        setup_dict = {
            'sample': setup_sample,
            'freq': setup_freq,
            'beat' : setup_beat
        }

        fig, ax = plt.subplots(len(setup_dict), 1)
        data['fig'] = fig

        for axis in ax:
            axis.set_axis_off()
        fig.set_facecolor('black')
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0, wspace=0)

        for i, (plot_type, setup_func) in enumerate(setup_dict.items()):
            setup_data = setup_func(ax[i])
            subfig[plot_type] = setup_data.pop('plot')
            data.update(setup_data)

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

    def setup_sample(axes):
        time_axis = [- i/sample_rate for i in range(sample_rate)[::-skip]]
        limits = (time_axis[0], time_axis[-1], audio.min(), nchannel - 1 + audio.max())
        axes.axis(limits)
        sample_buffer = np.array([[ch]*len(time_axis) for ch in range(nchannel)], dtype=np.float64)
        return {
            'plot'           : axes.plot(time_axis, sample_buffer.T),
            'sample_buffer' : sample_buffer
        }
    def update_sample(offset, shift):
        if offset >= nsample:
            return []

        if shift > sample_rate:
            offset += shift // sample_rate * sample_rate
            shift %= sample_rate
        dshift = (skip - 1 + shift) // skip
        data['sample_buffer'] = np.roll(data['sample_buffer'], -dshift, axis=1)

        for ch, (line, ch_sample) in enumerate(zip(subfig['sample'], data['sample_buffer'])):
            if offset + shift < nsample:
                ch_sample[-dshift:] = ch + audio[offset:offset + shift:skip, ch]
            else:
                ch_sample[-dshift:((nsample + skip - offset - 1) // skip) - dshift] = ch + audio[offset::skip, ch]

            line.set_ydata(ch_sample)

        return subfig['sample']

    def setup_freq(axes):
        axes.axis((0, 20000, -200, 200))
        return {'plot' : axes.plot([1], [[1, -1]])}
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

    def setup_beat(axes):
        axes.axis((0, n_bands, 0, 40))
        return {
            'plot' : axes.plot(range(n_bands), np.zeros((n_bands, nchannel))),
            'shift_buffer' : 0,
            'history_buffer' : np.zeros((nchannel, buffer_size, n_bands))
        }
    def update_beat(offset):
        shift_buffer = data['shift_buffer']
        if offset - shift_buffer < n_history or offset >= nsample:
            return []
        n_update = (offset - shift_buffer) // n_history
        data['shift_buffer'] += n_update*n_history
        data['history_buffer'] = np.roll(data['history_buffer'], -n_update, axis=0)
        energy_buffer = data['history_buffer']
        for ch, (line, ch_buff) in enumerate(zip(subfig['beat'], energy_buffer)):
            for i in range(n_update):
                if n_update - i >= n_history:
                    continue
                energy = np.abs(np.fft.rfft(audio[shift_buffer + i*n_history:shift_buffer + (i + 1)*n_history, ch]))**2
                ch_buff[i - n_update] = np.array([
                    (j + 1)/n_history * np.sum(energy[k:k + j + 1])
                    for j, k in enumerate(accumulate(range(n_bands)))
                ])
            line.set_ydata(ch_buff[-1])
        return subfig['beat']

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
