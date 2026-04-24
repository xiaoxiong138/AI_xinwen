import unittest

from src.notifier import verify_email_arrival


class EmailArrivalVerificationTests(unittest.TestCase):
    def test_verify_email_arrival_skips_when_config_is_missing(self):
        result = verify_email_arrival(
            imap_server="",
            imap_port=993,
            username="",
            password="",
            subject_contains="AI Frontier",
        )

        self.assertFalse(result["enabled"])
        self.assertEqual(result["status"], "skipped_missing_config")


if __name__ == "__main__":
    unittest.main()
