import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


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
