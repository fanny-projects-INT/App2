import unittest

from alyx_rest import load_raw_sessions


class FakeClient:
    def __init__(self):
        self.rest_schemes = type("Schemes", (), {"endpoints": ["sessions"]})()
        self.calls = []

    def rest(self, endpoint, action, **kwargs):
        self.calls.append((endpoint, action, kwargs))
        if (endpoint, action) == ("sessions", "list"):
            return [{"id": "session-id"}]
        if (endpoint, action) == ("sessions", "read"):
            return {
                "id": "session-id",
                "subject": "VF101",
                "start_time": "2026-09-30T10:00:00",
                "end_time": "2026-09-30T10:20:00",
                "number": 1,
                "task_protocol": "Task",
                "type": "Experiment",
                "n_correct_trials": 15,
                "n_trials": 20,
                "narrative": "Calm session.",
                "json": {"trials": {"valid": 18}},
                "location": "rig2",
                "users": ["experimenter"],
            }
        raise AssertionError((endpoint, action, kwargs))


class AlyxRestTests(unittest.TestCase):
    def test_loads_session_details_using_list_and_read_only(self):
        client = FakeClient()

        rows = load_raw_sessions(client, "cazetteslab")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mouse_id"], "VF101")
        self.assertEqual(rows[0]["experimentalists"], ["experimenter"])
        self.assertEqual(rows[0]["poop_actions"], [])
        self.assertEqual(
            [(endpoint, action) for endpoint, action, _ in client.calls],
            [("sessions", "list"), ("sessions", "read")],
        )


if __name__ == "__main__":
    unittest.main()
