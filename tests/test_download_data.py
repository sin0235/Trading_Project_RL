import unittest
from unittest.mock import patch

import pandas as pd

from src.data.download_data import DownloadData


class DownloadDataRateLimitTests(unittest.TestCase):
    def test_extract_retry_wait_seconds_from_vnstock_message(self):
        downloader = DownloadData(["FPT"], "2026-01-01", "2026-03-22")
        error = SystemExit("Rate limit exceeded. Chờ 57 giây để tiếp tục")

        wait_seconds = downloader._extract_retry_wait_seconds(error)

        self.assertEqual(wait_seconds, 57.0)

    def test_retry_after_rate_limit_then_succeeds(self):
        downloader = DownloadData(
            ["FPT"],
            "2026-01-01",
            "2026-03-22",
            delay=0.0,
            max_retries=1,
            retry_buffer_seconds=3.0,
        )
        calls = {"count": 0}

        def fake_get_single(_ticker):
            calls["count"] += 1
            if calls["count"] == 1:
                raise SystemExit("Rate limit exceeded. Chờ 5 giây để tiếp tục")
            return pd.DataFrame({"close": [1.0]})

        downloader._get_single = fake_get_single  # type: ignore[method-assign]
        downloader._ensure_registration = lambda: None  # type: ignore[method-assign]

        with patch("src.data.download_data.time.sleep") as sleep_mock:
            downloader.download_all()

        self.assertEqual(calls["count"], 2)
        self.assertIn("FPT", downloader.data)
        sleep_mock.assert_any_call(8.0)


if __name__ == "__main__":
    unittest.main()
