import imaplib
import smtplib
import time
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import decode_header
from email.utils import parsedate_to_datetime


class EmailNotifier:
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        sender_email: str,
        sender_password: str,
        timeout_seconds: int = 30,
        max_attempts: int = 3,
        retry_delay_seconds: int = 5,
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, int(max_attempts))
        self.retry_delay_seconds = max(0, int(retry_delay_seconds))

    def send_email(self, recipient_email: str, subject: str, html_content: str):
        if not self.sender_email or not self.sender_password:
            print("Warning: Email credentials not configured. Skipping email send.")
            return False

        for attempt in range(1, self.max_attempts + 1):
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                msg["From"] = self.sender_email
                msg["To"] = recipient_email
                msg.attach(MIMEText(html_content, "html", "utf-8"))
                print(
                    f"Connecting to SMTP server {self.smtp_server}:{self.smtp_port} "
                    f"(attempt {attempt}/{self.max_attempts})..."
                )
                if int(self.smtp_port) == 465:
                    with smtplib.SMTP_SSL(
                        self.smtp_server,
                        int(self.smtp_port),
                        timeout=self.timeout_seconds,
                    ) as server:
                        server.login(self.sender_email, self.sender_password)
                        server.sendmail(self.sender_email, recipient_email, msg.as_string())
                else:
                    with smtplib.SMTP(
                        self.smtp_server,
                        int(self.smtp_port),
                        timeout=self.timeout_seconds,
                    ) as server:
                        server.ehlo()
                        server.starttls()
                        server.ehlo()
                        server.login(self.sender_email, self.sender_password)
                        server.sendmail(self.sender_email, recipient_email, msg.as_string())
                print(f"Email sent successfully to {recipient_email}")
                return True
            except Exception as exc:
                print(f"SMTP Error on attempt {attempt}/{self.max_attempts}: {exc}")
                if attempt < self.max_attempts and self.retry_delay_seconds > 0:
                    time.sleep(self.retry_delay_seconds)
        return False


def _decode_header_text(value: str) -> str:
    parts = decode_header(value or "")
    decoded = []
    for payload, charset in parts:
        if isinstance(payload, bytes):
            decoded.append(payload.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(payload)
    return "".join(decoded)


def verify_email_arrival(
    *,
    imap_server: str,
    imap_port: int,
    username: str,
    password: str,
    subject_contains: str,
    since_minutes: int = 30,
    mailbox: str = "INBOX",
    timeout_seconds: int = 30,
) -> dict:
    if not imap_server or not username or not password or not subject_contains:
        return {
            "enabled": False,
            "verified": False,
            "status": "skipped_missing_config",
            "matched_subject": "",
            "matched_date": "",
            "error": "",
        }

    cutoff = datetime.now() - timedelta(minutes=max(1, int(since_minutes)))
    try:
        with imaplib.IMAP4_SSL(imap_server, int(imap_port), timeout=timeout_seconds) as client:
            client.login(username, password)
            client.select(mailbox)
            search_since = cutoff.strftime("%d-%b-%Y")
            status, data = client.search(None, "SINCE", search_since)
            if status != "OK":
                return {
                    "enabled": True,
                    "verified": False,
                    "status": "search_failed",
                    "matched_subject": "",
                    "matched_date": "",
                    "error": str(data),
                }
            message_ids = (data[0] or b"").split()
            for message_id in reversed(message_ids[-50:]):
                fetch_status, fetch_data = client.fetch(message_id, "(BODY.PEEK[HEADER.FIELDS (SUBJECT DATE)])")
                if fetch_status != "OK" or not fetch_data:
                    continue
                raw_header = fetch_data[0][1].decode("utf-8", errors="replace")
                subject = ""
                message_date = ""
                for line in raw_header.splitlines():
                    if line.lower().startswith("subject:"):
                        subject = _decode_header_text(line.split(":", 1)[1].strip())
                    elif line.lower().startswith("date:"):
                        message_date = line.split(":", 1)[1].strip()
                if subject_contains not in subject:
                    continue
                try:
                    parsed_date = parsedate_to_datetime(message_date)
                    if parsed_date.tzinfo is not None:
                        parsed_date = parsed_date.astimezone().replace(tzinfo=None)
                    if parsed_date < cutoff:
                        continue
                except Exception:
                    pass
                return {
                    "enabled": True,
                    "verified": True,
                    "status": "found",
                    "matched_subject": subject,
                    "matched_date": message_date,
                    "error": "",
                }
            return {
                "enabled": True,
                "verified": False,
                "status": "not_found",
                "matched_subject": "",
                "matched_date": "",
                "error": "",
            }
    except Exception as exc:
        return {
            "enabled": True,
            "verified": False,
            "status": "error",
            "matched_subject": "",
            "matched_date": "",
            "error": str(exc),
        }
