import unittest
import numpy as np
from dashboard.raw_qc import compute_maps, load_saved_rms, prepare_display_rms
import tempfile
from pathlib import Path


class RawQCTests(unittest.TestCase):
    def test_calibrated_sine_dc_removal_spectrum_and_sync_exclusion(self):
        class Reader:
            fs, ns, nc, nsync = 1000, 2000, 3, 1

            def read(self, samples, channels, sync=False):
                t = np.arange(self.ns) / self.fs
                wave = 10e-6 * np.sin(2 * np.pi * 50 * t)
                return np.column_stack((wave + .001, wave * 2, np.ones(self.ns)))[samples, channels]

        result = compute_maps(Reader(), windows=10, window_s=.2)
        self.assertEqual(result['rms'].shape, (2, 10))
        np.testing.assert_allclose(result['rms'][0], 10 / np.sqrt(2), rtol=1e-6)
        np.testing.assert_allclose(result['rms'][1], 20 / np.sqrt(2), rtol=1e-6)
        peak = result['frequencies'][np.argmax(result['psd'][0])]
        self.assertEqual(peak, 50)
        self.assertEqual(result['duration'], 2)

    def test_default_samples_64_windows_across_entire_duration(self):
        class Reader:
            fs, ns, nc, nsync = 100, 100000, 1, 0
            reads = []

            def read(self, samples, channels, sync=False):
                self.reads.append((samples.start, samples.stop))
                return np.zeros((samples.stop - samples.start, 1))

        reader = Reader()
        result = compute_maps(reader)
        self.assertEqual(result['rms'].shape, (1, 64))
        self.assertEqual(reader.reads[0][0], 0)
        self.assertEqual(reader.reads[-1][1], reader.ns)
        self.assertEqual(sum(stop - start for start, stop in reader.reads), 64 * 25)

    def test_saved_rms_keeps_values_and_transposes_time_channel_axes(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            values = np.arange(12.).reshape(3, 4)
            np.save(folder / '_iblqc_ephysTimeRmsAP.rms.npy', values)
            np.save(folder / '_iblqc_ephysTimeRmsAP.timestamps.npy', [1.5, 4.5, 7.5])
            times, loaded = load_saved_rms(str(folder), ())
            np.testing.assert_array_equal(loaded, values.T)
            np.testing.assert_array_equal(times, [1.5, 4.5, 7.5])

    def test_spectrum_only_skips_rms(self):
        class Reader:
            fs, ns, nc, nsync = 1000, 100, 1, 0
            def read(self, samples, channels, sync=False):
                return np.zeros((100, 1))[samples, channels]
        result = compute_maps(Reader(), include_rms=False)
        self.assertIsNone(result['rms'])
        self.assertEqual(result['psd'].shape, (1, 51))

    def test_display_pooling_retains_last_window_and_rms_energy(self):
        times, values = prepare_display_rms(np.arange(5.), np.array([[0., 0., 0., 0., 10.]]), max_bins=2)
        self.assertEqual(values.shape, (1, 2))
        self.assertEqual(values.dtype, np.float32)
        self.assertAlmostEqual(values[0, -1], np.sqrt(50), places=5)
        np.testing.assert_array_equal(times, [1., 3.5])

    def test_short_recording_windows_are_bounded(self):
        class Reader:
            fs, ns, nc, nsync = 1000, 20, 1, 0

            def read(self, samples, channels, sync=False):
                return np.zeros((self.ns, 1))[samples, channels]

        result = compute_maps(Reader(), windows=1)
        self.assertEqual(result['rms'].shape, (1, 1))
        self.assertEqual(result['window_s'], .02)


if __name__ == '__main__':
    unittest.main()
