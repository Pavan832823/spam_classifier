"""
Gmail Service — Fixed body extraction + sender parsing
"""

import imaplib
import email
from email.header import decode_header, make_header
from email import policy
import re

IMAP_SERVER   = "imap.gmail.com"
IMAP_PORT     = 993
INITIAL_LIMIT = 50


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\r\n', '\n', text)
    # Remove raw MIME boundary lines
    text = re.sub(r'--[A-Za-z0-9]+(\r?\n|$)', '', text)
    # Remove Content-Type headers that leaked into body
    text = re.sub(r'Content-Type:.*?(\r?\n|$)', '', text)
    text = re.sub(r'Content-Transfer-Encoding:.*?(\r?\n|$)', '', text)
    return text.strip()


def decode_str(s):
    """Safely decode encoded email header strings."""
    if not s:
        return ""
    try:
        return str(make_header(decode_header(s)))
    except Exception:
        if isinstance(s, bytes):
            return s.decode("utf-8", errors="ignore")
        return s


def extract_sender(msg):
    """Extract clean sender name/email from From header."""
    raw = msg.get("From", "")
    if not raw:
        return "Unknown"
    decoded = decode_str(raw)
    # Try to get display name: "John Doe <john@example.com>"
    match = re.match(r'^"?([^"<]+)"?\s*<', decoded)
    if match:
        name = match.group(1).strip()
        if name:
            return name
    # Fall back to email address
    email_match = re.search(r'[\w.+-]+@[\w.-]+\.\w+', decoded)
    if email_match:
        return email_match.group(0)
    return decoded.strip() or "Unknown"


def extract_body(msg):
    """
    Reliably extract plain-text body from email.Message.
    Handles multipart, HTML-only, and encoded emails.
    """
    plain_body = ""
    html_body  = ""

    if msg.is_multipart():
        for part in msg.walk():
            ct   = part.get_content_type()
            disp = str(part.get("Content-Disposition", ""))
            if "attachment" in disp:
                continue
            try:
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                charset = part.get_content_charset() or "utf-8"
                text    = payload.decode(charset, errors="ignore")
                if ct == "text/plain" and not plain_body:
                    plain_body = text
                elif ct == "text/html" and not html_body:
                    html_body  = text
            except Exception:
                continue
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                charset    = msg.get_content_charset() or "utf-8"
                plain_body = payload.decode(charset, errors="ignore")
        except Exception:
            plain_body = str(msg.get_payload())

    # Prefer plain text; fall back to stripped HTML
    body = plain_body or _strip_html(html_body)
    return clean_text(body[:5000])


def _strip_html(html):
    """Strip HTML tags to get readable text."""
    if not html:
        return ""
    text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<p[^>]*>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def fetch_new_emails(gmail_user, app_password, last_uid, fetch_all=False):
    """
    Fetch emails from Gmail via IMAP.

    Args:
        gmail_user:   Gmail address
        app_password: Gmail app password  
        last_uid:     Last processed UID (0 = first run)
        fetch_all:    If True, re-fetch all emails (ignores last_uid)

    Returns:
        (emails: list[dict], max_uid: int)
    """
    mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
    mail.login(gmail_user, app_password)
    mail.select("inbox")

    # Decide which UIDs to fetch
    if fetch_all or last_uid == 0:
        status, messages = mail.uid("search", None, "ALL")
        all_uids    = messages[0].split()
        email_uids  = all_uids[-INITIAL_LIMIT:]
    else:
        status, messages = mail.uid("search", None, f"UID {last_uid + 1}:*")
        email_uids = messages[0].split()

    emails  = []
    max_uid = last_uid

    for uid in email_uids:
        try:
            _, msg_data = mail.uid("fetch", uid, "(RFC822)")

            if not msg_data or not msg_data[0] or not isinstance(msg_data[0], tuple):
                continue

            raw = msg_data[0][1]
            if not raw:
                continue

            msg = email.message_from_bytes(raw)

            subject = decode_str(msg.get("Subject", "")) or "(no subject)"
            sender  = extract_sender(msg)
            body    = extract_body(msg)

            emails.append({
                "uid":     int(uid),
                "sender":  sender,
                "subject": subject,
                "body":    body,
            })

            max_uid = max(max_uid, int(uid))

        except Exception as e:
            print(f"⚠️  Skipping UID {uid}: {e}")
            continue

    mail.logout()
    return emails, max_uid
