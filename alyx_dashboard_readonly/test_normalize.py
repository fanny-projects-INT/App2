import unittest

from normalize import _probabilities, session_to_row


class NormalizeSessionTests(unittest.TestCase):
    def test_flattens_current_gui_session_and_derives_bout_rate(self):
        row = session_to_row({
            "session_id": "session-id",
            "mouse_id": "VF101",
            "start_time": "2026-09-30T10:00:00",
            "end_time": "2026-09-30T10:20:00",
            "session_number": 2,
            "protocol": "Task",
            "session_type": "Experiment",
            "n_correct_trials": 15,
            "n_trials": 20,
            "narrative": "Calm session.",
            "location": "rig2",
            "experimentalists": ["experimenter"],
            "poop_actions": [{"narrative": "3 poop samples collected.", "json": {}}],
            "session_json": {
                "trials": {"total": 20, "valid": 18, "correct": 15},
                "reward_rate": 0.6,
                "water_administration": {"manual": 2, "reward": 12, "total": 14},
                "virtual_foraging": {
                    "recording_type": "2",
                    "reward_probability": "0.9",
                    "switch_probability": "0.1",
                    "rig_number": "2",
                },
            },
        })
        self.assertEqual(row["date"], "2026-09-30")
        self.assertEqual(row["experimentalist"], "experimenter")
        self.assertEqual(row["recording_type"], "Neuropixel")
        self.assertEqual(row["training_type"], "Task")
        self.assertEqual(row["session_probabilities"], "90/10")
        self.assertEqual(row["valid_bouts"], 18.0)
        self.assertEqual(row["poops_number"], 3.0)
        self.assertEqual(row["bout_rate_per_min"], 1.0)
        self.assertEqual(row["session_comments"], "Calm session.")

    def test_missing_historical_fields_remain_empty(self):
        row = session_to_row({
            "mouse_id": "VF074",
            "protocol": "OFT",
            "session_json": {},
            "experimentalists": [],
        })
        self.assertIsNone(row["licks"])
        self.assertIsNone(row["tube"])
        self.assertIsNone(row["headfix"])
        self.assertIsNone(row["comfort_on_treadmill"])

    def test_probability_formats_are_consistent(self):
        self.assertEqual(
            _probabilities({}, {"reward_probability": 0.9, "switch_probability": 0.3}),
            "90/30",
        )
        self.assertEqual(_probabilities({}, {"probas": "90/30; 45/15"}), "90/30 · 45/15")
        self.assertEqual(_probabilities({}, {"probas": "90/00 -blue"}), "90/00 · Blue")


if __name__ == "__main__":
    unittest.main()

