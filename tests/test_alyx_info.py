import unittest
from unittest.mock import patch

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
    def test_secrets_login_uses_server_account_without_logout(self):
        app = AppTest.from_string('from alyx_info import render_info_view\nrender_info_view("VF101")')
        app.secrets["alyx"] = {"username": "configured-user", "password": "test-password"}
        with patch("alyx_info.connect", return_value=Client()) as connect:
            app.run()
            self.assertFalse(app.exception)
            connect.assert_called_once_with("https://cazettes.iblcore.org", "configured-user", "test-password")
            self.assertEqual(app.session_state["info_alyx_data"]["mouse"], "VF101")
            self.assertNotIn("info_alyx_logout", [button.key for button in app.button])

    def test_requests_only_selected_mouse(self):
        client = Client()
        rows = load_raw_sessions(client, subject="VF101")
        self.assertEqual(rows[0]["mouse_id"], "VF101")
        self.assertEqual(client.calls[0][2]["subject"], "VF101")
        self.assertTrue(all(action in {"read", "list"} for _, action, _ in client.calls))

    def test_login_and_switch_mouse(self):
        app = AppTest.from_string('from alyx_info import render_info_view\nimport streamlit as st\nrender_info_view(st.session_state.get("mouse", "VF101"))')
        client = Client()
        with patch("alyx_info._secret_credentials", return_value=("", "")), \
                patch("alyx_info.connect", return_value=client):
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
