import unittest

import numpy as np
import pandas as pd

from dashboard.probe_profile import prepare_profile, build_profile_figure


class ProbeProfileTests(unittest.TestCase):
    def setUp(self):
        # Two contacts at each depth must not double the sampled region span.
        self.channels = pd.DataFrame({
            'depth_um': [0, 0, 20, 20, 40, 40, 60, 60],
            'brain_region': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        })
        self.units = pd.DataFrame({
            'unit_id': [0, 1, 2], 'bombcell_label': ['good', 'good', 'noise'],
            'depth_um': [0, 20, 40], 'brain_region': ['A', 'A', 'B'],
            'firing_rate': [2., 4., 100.], 'amplitude_median': [-10., -20., -100.],
        })

    def test_small_and_zero_unit_regions_remain_with_correct_density(self):
        segments, good, summary = prepare_profile(self.channels, self.units)
        rows = summary.set_index('brain_region')
        self.assertEqual(set(rows.index), {'A'})
        self.assertEqual(rows.loc['A', 'channels'], 4)
        self.assertAlmostEqual(rows.loc['A', 'span_mm'], .04)
        self.assertAlmostEqual(rows.loc['A', 'density'], 50.)
        self.assertEqual(rows.loc['A', 'firing_rate'], 3.)
        self.assertEqual(rows.loc['A', 'amplitude'], 15.)
        figure = build_profile_figure(segments, good, lambda _: '#123456')
        self.assertEqual(len(figure.data), 3)
        self.assertEqual(figure.data[1].marker.color, '#D5DCE3')
        self.assertEqual(len(figure.layout.shapes), 1)
        self.assertEqual(figure.layout.shapes[0].x0, 3.)
        self.assertEqual(figure.layout.shapes[0].y0, -10.)
        self.assertEqual(figure.layout.shapes[0].y1, 30.)

    def test_region_hover_contains_summary_information(self):
        segments, good, summary = prepare_profile(self.channels, self.units)
        figure = build_profile_figure(segments, good, lambda _: '#123456', region_summary=summary)
        hover = figure.data[0].hovertemplate
        for text in ('Good units: 2', 'Channels: 4', 'Span (mm): 0.04',
                     'Good units / mm: 50.0', 'Median rate (Hz): 3.00', '15.0'):
            self.assertIn(text, hover)
        self.assertIn('Good units: 0', figure.data[1].hovertemplate)

    def test_amplitude_selector_updates_points_and_median(self):
        segments, good, _ = prepare_profile(self.channels, self.units)
        figure = build_profile_figure(segments, good, lambda _: '#123456', metric='amplitude_median')
        self.assertEqual(list(figure.data[-1].x), [10., 20.])
        self.assertEqual(figure.layout.shapes[0].x0, 15.)
        self.assertIn('Amplitude', figure.layout.xaxis2.title.text)

    def test_count_metric_uses_region_totals_and_omits_empty_regions(self):
        segments, good, _ = prepare_profile(self.channels, self.units)
        figure = build_profile_figure(segments, good, lambda _: '#123456', metric='good_units')
        self.assertEqual(len(figure.data), 3)
        self.assertEqual(figure.data[-1].name, 'A')
        self.assertEqual(list(figure.data[-1].x), [2])
        self.assertEqual(len(figure.layout.shapes), 0)

    def test_no_good_units_preserves_region_map(self):
        self.units['bombcell_label'] = 'noise'
        segments, good, summary = prepare_profile(self.channels, self.units)
        self.assertTrue(good.empty)
        self.assertEqual(summary.good_units.sum(), 0)
        self.assertTrue(summary.empty)
        figure = build_profile_figure(segments, good, lambda _: '#123456')
        self.assertEqual(len(figure.data), 2)

    def test_empty_channel_map_has_no_figure(self):
        segments, good, summary = prepare_profile(self.channels.iloc[:0], self.units)
        self.assertIsNone(build_profile_figure(segments, good, lambda _: '#123456'))
        self.assertEqual(summary.good_units.sum(), 2)


if __name__ == '__main__':
    unittest.main()
