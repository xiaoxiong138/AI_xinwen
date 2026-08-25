import unittest
from datetime import datetime, timedelta
from email.message import EmailMessage
from email.utils import format_datetime

from src.notifier import _subject_matches, verify_email_arrival


class FakeIMAP:
    def __init__(self, *args, **kwargs):
        message = EmailMessage()
        message["Subject"] = "AI Frontier Intelligence Daily"
        message["Date"] = format_datetime(datetime.now().astimezone() - timedelta(minutes=1))
        message["From"] = "sender@example.com"
        self.raw_header = message.as_bytes()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def login(self, username, password):
        return "OK", []

    def select(self, mailbox):
        return "OK", []

    def search(self, *args):
        return "OK", [b"1"]

    def fetch(self, message_id, query):
        return "OK", [(b"1", self.raw_header)]


class MailboxFallbackIMAP(FakeIMAP):
    selected_mailboxes = []

    def select(self, mailbox):
        self.__class__.selected_mailboxes.append(mailbox)
        if mailbox == "INBOX":
            return "NO", [b"missing"]
        return "OK", []


class RetryIMAP(FakeIMAP):
    search_calls = 0

    def search(self, *args):
        self.__class__.search_calls += 1
        if self.__class__.search_calls == 1:
            return "OK", [b""]
        return "OK", [b"1"]


class EmailArrivalVerificationTests(unittest.TestCase):
    def test_subject_match_rejects_different_report_timestamps(self):
        self.assertFalse(
            _subject_matches(
                "[2026-07-16 22:02] AI Frontier Intelligence Daily - 21:00 Resend",
                "[2026-07-16 13:38] AI Frontier Intelligence Daily - 13:00 Resend",
            )
        )

    def test_subject_match_rejects_custom_test_prefix_for_official_report(self):
        self.assertFalse(
            _subject_matches(
                "[2026-07-31 14:14] AI Frontier Intelligence Daily",
                "[V10 PLAIN SUMMARY 2026-07-31 03:20] AI Frontier Intelligence Daily",
            )
        )

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

    def test_verify_email_arrival_matches_subject_without_timestamp_prefix(self):
        import src.notifier as notifier

        original_imap = notifier.imaplib.IMAP4_SSL
        notifier.imaplib.IMAP4_SSL = FakeIMAP
        try:
            result = verify_email_arrival(
                imap_server="imap.example.com",
                imap_port=993,
                username="sender@example.com",
                password="password",
                subject_contains="[2026-05-26 21:02] AI Frontier Intelligence Daily",
                since_minutes=1440,
                expected_sender="sender@example.com",
            )
        finally:
            notifier.imaplib.IMAP4_SSL = original_imap

        self.assertTrue(result["verified"])
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["matched_from"], "sender@example.com")
        self.assertEqual(result["checked_count"], 1)

    def test_verify_email_arrival_checks_fallback_mailboxes(self):
        import src.notifier as notifier

        original_imap = notifier.imaplib.IMAP4_SSL
        MailboxFallbackIMAP.selected_mailboxes = []
        notifier.imaplib.IMAP4_SSL = MailboxFallbackIMAP
        try:
            result = verify_email_arrival(
                imap_server="imap.example.com",
                imap_port=993,
                username="sender@example.com",
                password="password",
                subject_contains="AI Frontier Intelligence Daily",
                mailbox=["INBOX", "Archive"],
                expected_sender="sender@example.com",
            )
        finally:
            notifier.imaplib.IMAP4_SSL = original_imap

        self.assertTrue(result["verified"])
        self.assertEqual(result["matched_mailbox"], "Archive")
        self.assertEqual(MailboxFallbackIMAP.selected_mailboxes, ["INBOX", "Archive"])

    def test_verify_email_arrival_retries_transient_not_found(self):
        import src.notifier as notifier

        original_imap = notifier.imaplib.IMAP4_SSL
        RetryIMAP.search_calls = 0
        notifier.imaplib.IMAP4_SSL = RetryIMAP
        try:
            result = verify_email_arrival(
                imap_server="imap.example.com",
                imap_port=993,
                username="sender@example.com",
                password="password",
                subject_contains="AI Frontier Intelligence Daily",
                expected_sender="sender@example.com",
                retry_attempts=2,
                retry_delay_seconds=0,
            )
        finally:
            notifier.imaplib.IMAP4_SSL = original_imap

        self.assertTrue(result["verified"])
        self.assertEqual(result["status"], "found")
        self.assertEqual(RetryIMAP.search_calls, 2)


if __name__ == "__main__":
    unittest.main()
