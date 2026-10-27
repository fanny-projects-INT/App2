import unittest
from threading import Event

from dashboard.ui_states import DataPrefetcher


class LoadingTests(unittest.TestCase):
    def test_prefetch_deduplicates_and_bounds_pending_work(self):
        worker = DataPrefetcher()
        started, release = Event(), Event()

        def work():
            started.set()
            release.wait(3)

        try:
            worker.submit("mouse-1", work)
            self.assertTrue(started.wait(2))
            worker.submit("mouse-1", work)
            worker.submit("mouse-2", work)
            worker.submit("mouse-3", work)
            self.assertEqual(set(worker.jobs), {"mouse-1", "mouse-2"})
        finally:
            release.set()
            worker.executor.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
