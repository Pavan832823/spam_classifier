import imaplib
import email
from email.header import decode_header


class IMAPService:

    def __init__(self, email_address, app_password):
        self.email_address = email_address
        self.app_password = app_password
        self.mail = imaplib.IMAP4_SSL("imap.gmail.com")
        self.mail.login(self.email_address, self.app_password)

    def fetch_recent_emails(self, max_results=5):
        self.mail.select("inbox")

        status, messages = self.mail.search(None, "ALL")
        email_ids = messages[0].split()

        emails = []

        for eid in email_ids[-max_results:]:
            _, msg_data = self.mail.fetch(eid, "(RFC822)")
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)

            # Subject
            subject, encoding = decode_header(msg["Subject"])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(errors="ignore")

            # Sender
            sender = msg.get("From")

            # Body extraction
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode(errors="ignore")
                        break
            else:
                body = msg.get_payload(decode=True).decode(errors="ignore")

            emails.append({
                "sender": sender,
                "subject": subject,
                "body": body[:1000]
            })

        return emails
