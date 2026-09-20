import imaplib
import re
import smtplib
import time
from datetime import datetime, timedelta
from email import policy
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import decode_header
from email.parser import BytesParser
from email.utils import parseaddr, parsedate_to_datetime


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


def _normalize_match_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _subject_match_variants(subject: str) -> list[str]:
    normalized = str(subject or "").strip()
    variants = [normalized]
    without_timestamp = re.sub(r"^\[[^\]]*\]\s*", "", normalized).strip()
    if without_timestamp and without_timestamp not in variants:
        variants.append(without_timestamp)
    for marker in ("AI Frontier Intelligence Daily", "AI日报", "AI 日报"):
        if marker.lower() in normalized.lower() and marker not in variants:
            variants.append(marker)
    return [variant for variant in variants if variant]


def _subject_matches(expected: str, actual: str) -> bool:
    timestamp_pattern = r"^\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\]"
    expected_timestamp = re.match(timestamp_pattern, str(expected or "").strip())
    actual_timestamp = re.match(timestamp_pattern, str(actual or "").strip())
    if expected_timestamp and actual_timestamp and expected_timestamp.group(1) != actual_timestamp.group(1):
        return False
    if expected_timestamp and not actual_timestamp and str(actual or "").lstrip().startswith("["):
        return False
    expected_volume = re.search(r"\[(\d+)\s*/\s*(\d+)\]", str(expected or ""))
    if expected_volume:
        actual_volume = re.search(r"\[(\d+)\s*/\s*(\d+)\]", str(actual or ""))
        if not actual_volume or actual_volume.groups() != expected_volume.groups():
            return False
    actual_norm = _normalize_match_text(actual)
    if not actual_norm:
        return False
    for variant in _subject_match_variants(expected):
        variant_norm = _normalize_match_text(variant)
        if variant_norm and (variant_norm in actual_norm or actual_norm in variant_norm):
            return True
        tokens = [token for token in re.split(r"\W+", variant_norm) if len(token) >= 3]
        if len(tokens) >= 3 and sum(1 for token in tokens if token in actual_norm) >= min(3, len(tokens)):
            return True
    return False


def _parse_header_bytes(raw_header: bytes) -> dict:
    message = BytesParser(policy=policy.default).parsebytes(raw_header or b"")
    return {
        "subject": _decode_header_text(str(message.get("Subject", ""))),
        "date": str(message.get("Date", "")),
        "from": _decode_header_text(str(message.get("From", ""))),
        "to": _decode_header_text(str(message.get("To", ""))),
    }


def _message_is_recent(message_date: str, cutoff: datetime) -> bool:
    try:
        parsed_date = parsedate_to_datetime(message_date)
        if parsed_date.tzinfo is not None:
            parsed_date = parsed_date.astimezone().replace(tzinfo=None)
        return parsed_date >= cutoff
    except Exception:
        return True


def _sender_matches(expected_sender: str, actual_from: str) -> bool:
    expected = _normalize_match_text(parseaddr(expected_sender or "")[1] or expected_sender)
    actual = _normalize_match_text(parseaddr(actual_from or "")[1] or actual_from)
    return not expected or not actual or expected == actual


def resolve_imap_server(smtp_server: str = "", configured_imap_server: str = "") -> str:
    configured = str(configured_imap_server or "").strip()
    if configured:
        return configured

    smtp = str(smtp_server or "").strip().lower()
    known_servers = {
        "smtp.gmail.com": "imap.gmail.com",
        "smtp.qq.com": "imap.qq.com",
        "smtp.163.com": "imap.163.com",
        "smtp.126.com": "imap.126.com",
        "smtp.office365.com": "outlook.office365.com",
    }
    if smtp in known_servers:
        return known_servers[smtp]
    if smtp.startswith("smtp."):
        return f"imap.{smtp[5:]}"
    return ""


def verify_email_arrival(
    *,
    imap_server: str,
    imap_port: int,
    username: str,
    password: str,
    subject_contains: str,
    since_minutes: int = 30,
    mailbox: str | list[str] = "INBOX",
    timeout_seconds: int = 30,
    expected_sender: str = "",
    max_messages: int = 120,
    retry_attempts: int = 1,
    retry_delay_seconds: int = 5,
) -> dict:
    if not imap_server or not username or not password or not subject_contains:
        return {
            "enabled": False,
            "verified": False,
            "status": "skipped_missing_config",
            "matched_subject": "",
            "matched_date": "",
            "matched_from": "",
            "matched_mailbox": "",
            "checked_count": 0,
            "error": "",
        }

    cutoff = datetime.now() - timedelta(minutes=max(1, int(since_minutes)))
    if isinstance(mailbox, (list, tuple)):
        mailboxes = [str(value or "").strip() for value in mailbox if str(value or "").strip()]
    else:
        raw_mailbox = str(mailbox or "INBOX")
        mailboxes = [part.strip() for part in re.split(r"[,;]", raw_mailbox) if part.strip()]
    mailboxes = mailboxes or ["INBOX"]

    checked_count = 0
    select_errors: list[str] = []
    attempts = max(1, int(retry_attempts or 1))
    try:
        for attempt in range(1, attempts + 1):
            with imaplib.IMAP4_SSL(imap_server, int(imap_port), timeout=timeout_seconds) as client:
                client.login(username, password)
                search_since = cutoff.strftime("%d-%b-%Y")
                for mailbox_name in mailboxes:
                    select_status, select_data = client.select(mailbox_name)
                    if select_status != "OK":
                        select_errors.append(f"{mailbox_name}: {select_data}")
                        continue
                    status, data = client.search(None, "SINCE", search_since)
                    if status != "OK":
                        return {
                            "enabled": True,
                            "verified": False,
                            "status": "search_failed",
                            "matched_subject": "",
                            "matched_date": "",
                            "matched_from": "",
                            "matched_mailbox": mailbox_name,
                            "checked_count": checked_count,
                            "error": str(data),
                        }
                    message_ids = (data[0] or b"").split()
                    for message_id in reversed(message_ids[-max(1, int(max_messages)):]):
                        fetch_status, fetch_data = client.fetch(message_id, "(BODY.PEEK[HEADER.FIELDS (SUBJECT DATE FROM TO)])")
                        if fetch_status != "OK" or not fetch_data:
                            continue
                        raw_payload = fetch_data[0][1] if isinstance(fetch_data[0], tuple) else b""
                        headers = _parse_header_bytes(raw_payload)
                        subject = headers["subject"]
                        message_date = headers["date"]
                        message_from = headers["from"]
                        checked_count += 1
                        if not _subject_matches(subject_contains, subject):
                            continue
                        if not _sender_matches(expected_sender, message_from):
                            continue
                        if not _message_is_recent(message_date, cutoff):
                            continue
                        return {
                            "enabled": True,
                            "verified": True,
                            "status": "found",
                            "matched_subject": subject,
                            "matched_date": message_date,
                            "matched_from": message_from,
                            "matched_mailbox": mailbox_name,
                            "checked_count": checked_count,
                            "error": "",
                        }
            if attempt < attempts and retry_delay_seconds > 0:
                time.sleep(max(0, int(retry_delay_seconds)))
        return {
            "enabled": True,
            "verified": False,
            "status": "not_found",
            "matched_subject": "",
            "matched_date": "",
            "matched_from": "",
            "matched_mailbox": "",
            "checked_count": checked_count,
            "error": "; ".join(select_errors[-5:]),
        }
    except Exception as exc:
        return {
            "enabled": True,
            "verified": False,
            "status": "error",
            "matched_subject": "",
            "matched_date": "",
            "matched_from": "",
            "matched_mailbox": "",
            "checked_count": checked_count,
            "error": str(exc),
        }
