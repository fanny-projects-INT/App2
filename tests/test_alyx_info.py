import unittest
from unittest.mock import patch
import pandas as pd

from streamlit.testing.v1 import AppTest

from alyx_dashboard_readonly.alyx_rest import load_raw_sessions


class Client:
    rest_schemes = type("Schemes", (), {"endpoints": ["sessions"]})()

    def __init__(self):
        self.calls = []

    def rest(self, endpoint, action, **kwargs):
        self.calls.append((endpoint, action, kwargs))
        if endpoint == "subjects":
            return {"nickname": kwargs["id"], "sex": "F"}
        if action == "list":
            return [{"id": kwargs["subject"]}]
        return {"id": kwargs["id"], "subject": kwargs["id"], "json": {},
                "start_time": "2026-09-30T10:00:00"}


class AlyxInfoTests(unittest.TestCase):
    def setUp(self):
        from dashboard.alyx_info import _configured_info_store
        _configured_info_store.clear()
        # Keep preloading deterministic in UI tests; asynchronous behavior is tested separately.
        worker = patch("dashboard.alyx_info.data_prefetcher")
        self.worker = worker.start().return_value
        self.worker.submit.side_effect = lambda key, job: job()
        self.addCleanup(worker.stop)

    def test_session_arrow_keeps_sessions_tab_and_reuses_loaded_data(self):
        app = AppTest.from_string('from dashboard.alyx_info import render_info_view\nrender_info_view("VF101", __import__("pandas").DataFrame({"Mouse_ID":[f"VF{i}" for i in range(100,108)], "Date":["2026-01-01"]*8}))')
        client = Client()
        with patch("dashboard.alyx_info._secret_credentials", return_value=("user", "password")), \
                patch("dashboard.alyx_info.connect", return_value=client):
            app.run()
            self.assertFalse(app.exception)
            calls = list(client.calls)
            app.session_state["info_alyx_tabs"] = "Sessions"
            app.run()
            app.button(key="toggle_alyx_session_0").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["info_alyx_tabs"], "Sessions")
            self.assertEqual(app.session_state["info_alyx_expanded_session"], "VF101")
            app.button(key="toggle_alyx_session_0").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["info_alyx_tabs"], "Sessions")
            self.assertIsNone(app.session_state["info_alyx_expanded_session"])
            self.assertEqual(client.calls, calls)

    def test_behavior_link_matches_version_and_rejects_ambiguous_sessions(self):
        from dashboard.alyx_info import _behavior_session_target
        local = pd.DataFrame({
            "Mouse_ID": ["VF101", "VF101"],
            "Date": pd.to_datetime(["2026-09-30", "2026-09-30"]),
            "Version": [1, 2],
        })
        row = {"mouse_id": "VF101", "date": "2026-09-30", "version": 2}
        self.assertEqual(_behavior_session_target(row, local), ("VF101", "2026-09-30", "2"))
        self.assertIsNone(_behavior_session_target(row, pd.concat([local, local])))

    def test_local_behavior_metrics_fill_missing_alyx_counts(self):
        from dashboard.alyx_info import _enrich_behavior_counts
        alyx = pd.DataFrame({
            "mouse_id": ["VF101"], "date": ["2026-09-30"], "version": [2.0],
            "valid_bouts": [None], "rewarded_licks": [None], "unrewarded_licks": [None],
        })
        local = pd.DataFrame({
            "Mouse_ID": ["VF101"], "Date": [pd.Timestamp("2026-09-30")], "Version": ["2"],
            "valid_bouts": [18], "rewarded_licks": [42], "unrewarded_licks": [9],
        })
        result = _enrich_behavior_counts(alyx, local, "VF101").iloc[0]
        self.assertEqual(result["valid_bouts"], 18)
        self.assertEqual(result["rewarded_licks"], 42)
        self.assertEqual(result["unrewarded_licks"], 9)

    def test_secrets_login_uses_server_account_without_logout(self):
        app = AppTest.from_string('from dashboard.alyx_info import render_info_view\nrender_info_view("VF101", __import__("pandas").DataFrame({"Mouse_ID":[f"VF{i}" for i in range(100,108)], "Date":["2026-01-01"]*8}))')
        app.secrets["alyx"] = {"username": "configured-user", "password": "test-password"}
        with patch("dashboard.alyx_info.connect", return_value=Client()) as connect:
            app.run()
            self.assertFalse(app.exception)
            connect.assert_called_once_with("https://cazettes.iblcore.org", "configured-user", "test-password")
            self.assertEqual(app.session_state["info_alyx_data"]["mouse"], "VF101")
            self.assertNotIn("info_alyx_logout", [button.key for button in app.button])

    def test_only_eight_recent_mice_cached_regardless_of_visits(self):
        app = AppTest.from_string('from dashboard.alyx_info import render_info_view\nimport streamlit as st\nrender_info_view(st.session_state.get("mouse", "VF100"), __import__("pandas").DataFrame({"Mouse_ID":[f"VF{i}" for i in range(100,108)], "Date":["2026-01-01"]*8}))')
        client = Client()
        with patch("dashboard.alyx_info._secret_credentials", return_value=("user", "password")), \
                patch("dashboard.alyx_info.connect", return_value=client) as connection:
            for i in range(8):
                app.session_state["mouse"] = f"VF{100+i}"
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.session_state["info_alyx_mouse_cache"]), 8)
            self.assertEqual(len(app.session_state["info_alyx_mouse_cache"]), 8)
            calls = len(client.calls)
            app.session_state["mouse"] = "VF100"
            app.run()
            self.assertEqual(len(client.calls), calls)
            app.session_state["mouse"] = "VF108"
            app.run()
            self.assertFalse(app.exception)
            cache = app.session_state["info_alyx_mouse_cache"]
            self.assertEqual(len(cache), 8)
            self.assertIn("VF100", cache)
            self.assertIn("VF101", cache)
            self.assertNotIn("VF108", cache)
            connection.assert_called_once()

    def test_cache_does_not_expire_and_refresh_button_is_absent(self):
        app = AppTest.from_string('from dashboard.alyx_info import render_info_view\nimport streamlit as st\nrender_info_view(st.session_state.get("mouse", "VF101"), __import__("pandas").DataFrame({"Mouse_ID":[f"VF{i}" for i in range(100,108)], "Date":["2026-01-01"]*8}))')
        client = Client()
        with patch("dashboard.alyx_info._secret_credentials", return_value=("user", "password")), \
                patch("dashboard.alyx_info.connect", return_value=client) as connection, \
                patch("dashboard.alyx_info.monotonic", return_value=0) as clock:
            app.run()
            calls = len(client.calls)
            clock.return_value = 301
            app.run()
            self.assertEqual(len(client.calls), calls)
            clock.return_value = 86400
            app.run()
            self.assertEqual(len(client.calls), calls)
            app.session_state["mouse"] = "VF102"
            app.run()
            other = app.session_state["info_alyx_mouse_cache"]["VF101"]
            calls = len(client.calls)
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(client.calls), calls)
            self.assertNotIn("info_alyx_refresh", [button.key for button in app.button])
            self.assertIs(app.session_state["info_alyx_mouse_cache"]["VF101"], other)
            connection.assert_called_once()

    def test_recent_mice_are_pinned_and_account_caches_are_isolated(self):
        from dashboard.alyx_info import InfoMouseCache, _recent_info_mice
        catalog = pd.DataFrame({"Mouse_ID":[f"VF{i}" for i in range(100, 110)],
                                "Date":pd.date_range("2026-01-01", periods=10)})
        recent = _recent_info_mice(catalog)
        self.assertEqual(recent, tuple(f"VF{i}" for i in range(109, 101, -1)))
        client = Client()
        store = InfoMouseCache(client)
        store.set_recent(recent)
        store.warm(recent)
        pinned = store.get("VF109")
        for i in range(20): store.get(f"older-{i}")
        self.assertEqual(len(store.entries), 8)
        calls = len(client.calls)
        self.assertIs(store.get("VF109"), pinned)
        self.assertEqual(len(client.calls), calls)
        self.assertTrue(all(mouse in store.entries for mouse in recent))
        other = InfoMouseCache(Client())
        self.assertEqual(other.entries, {})
        self.assertIsNot(other.get("VF109"), pinned)

    def test_server_account_reuses_cache_across_browser_sessions(self):
        from dashboard.alyx_info import _server_info_store
        client = Client()
        with patch("dashboard.alyx_info.connect", return_value=client) as connection:
            first = _server_info_store("server-user", "secret")
            first.set_recent(("VF101",))
            cached = first.get("VF101")
            calls = len(client.calls)
            second = _server_info_store("server-user", "secret")
            self.assertIs(first, second)
            self.assertIs(second.get("VF101"), cached)
            self.assertEqual(len(client.calls), calls)
            connection.assert_called_once()
            other = _server_info_store("other-user", "secret")
            self.assertIsNot(other, first)

    def test_hot_reload_migrates_old_cache_without_reconnecting(self):
        from threading import RLock
        from dashboard.alyx_info import InfoMouseCache, _upgrade_info_store, INFO_CACHE_VERSION
        class OldCache:
            def warm(self, mice): pass
        old = OldCache()
        old.client = Client()
        old.lock = RLock()
        old.pinned = ("VF101",)
        old.entries = {"VF101": {"mouse":"VF101", "version":INFO_CACHE_VERSION, "rows":[], "subject":{}}}
        entries = old.entries
        client = old.client
        migrated = _upgrade_info_store(old)
        self.assertIs(migrated, old)
        self.assertIsInstance(migrated, InfoMouseCache)
        self.assertIs(migrated.entries, entries)
        self.assertIs(migrated.client, client)
        progress = []
        migrated.warm(("VF101",), on_progress=progress.append)
        migrated.get("VF101", on_progress=progress.append)
        self.assertEqual(progress, [])
        self.assertEqual(client.calls, [])

    def test_info_renders_selected_mouse_before_background_preload(self):
        self.worker.submit.side_effect = None
        client = Client()
        app = AppTest.from_string('from dashboard.alyx_info import render_info_view\nimport pandas as pd\nrender_info_view("VF101", pd.DataFrame({"Mouse_ID":[f"VF{i}" for i in range(100,108)], "Date":["2026-01-01"]*8}))')
        with patch("dashboard.alyx_info._secret_credentials", return_value=("user", "password")), patch("dashboard.alyx_info.connect", return_value=client):
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(set(app.session_state["info_alyx_mouse_cache"]), {"VF101"})
            self.worker.submit.assert_called_once()
            job = self.worker.submit.call_args.args[1]
            job()
            self.assertEqual(len(app.session_state["info_alyx_mouse_cache"]), 8)

    def test_cached_mouse_is_available_during_network_preload(self):
        from dashboard.alyx_info import InfoMouseCache
        from concurrent.futures import ThreadPoolExecutor
        store = InfoMouseCache(Client())
        store.set_recent(("VF101",))
        cached = store.get("VF101")
        with ThreadPoolExecutor(max_workers=1) as pool:
            with store._request_slot(True):
                # A cached read must not wait for the Alyx request slot.
                self.assertIs(pool.submit(store.get, "VF101").result(timeout=1), cached)

    def test_requests_only_selected_mouse(self):
        client = Client()
        rows = load_raw_sessions(client, subject="VF101")
        self.assertEqual(rows[0]["mouse_id"], "VF101")
        self.assertEqual(client.calls[0][2]["subject"], "VF101")
        self.assertTrue(all(action in {"read", "list"} for _, action, _ in client.calls))

    def test_login_and_switch_mouse(self):
        app = AppTest.from_string('from dashboard.alyx_info import render_info_view\nimport streamlit as st\nrender_info_view(st.session_state.get("mouse", "VF101"), __import__("pandas").DataFrame({"Mouse_ID":[f"VF{i}" for i in range(100,108)], "Date":["2026-01-01"]*8}))')
        client = Client()
        with patch("dashboard.alyx_info._secret_credentials", return_value=("", "")), \
                patch("dashboard.alyx_info.connect", return_value=client):
            app.run()
            self.assertFalse(app.exception)
            app.text_input[0].input("user")
            app.text_input[1].input("secret")
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertNotIn("info_alyx_password", app.session_state)
            self.assertEqual(app.session_state["info_alyx_data"]["mouse"], "VF101")
            app.session_state["mouse"] = "VF102"
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["info_alyx_data"]["rows"][0]["mouse_id"], "VF102")


if __name__ == "__main__":
    unittest.main()
