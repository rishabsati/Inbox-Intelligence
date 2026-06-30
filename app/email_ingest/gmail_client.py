"""
Real Gmail ingestion via OAuth. This is the "impressive" data source --
it requires a one-time setup in Google Cloud Console (see README.md),
but after that, fetch_recent_emails() works just like sample_data's
get_sample_emails(), returning the same shape of dict.

First run will open a browser window for the user to grant read-only
access to their Gmail. The resulting token is cached to GMAIL_TOKEN_PATH
so they don't need to re-auth every time.
"""
import base64
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from app.config import settings

# Read-only scope is intentional -- this app never needs to send or modify mail.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def _get_credentials() -> Credentials:
    creds = None
    if os.path.exists(settings.GMAIL_TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(settings.GMAIL_TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(settings.GMAIL_CREDENTIALS_PATH):
                raise RuntimeError(
                    "Gmail credentials.json not found. Create an OAuth client ID "
                    "(Desktop app type) in Google Cloud Console, download the JSON, "
                    f"and save it to {settings.GMAIL_CREDENTIALS_PATH}. See README.md."
                )
            flow = InstalledAppFlow.from_client_secrets_file(settings.GMAIL_CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(settings.GMAIL_TOKEN_PATH, "w") as token_file:
            token_file.write(creds.to_json())

    return creds


def _html_to_text(html: str) -> str:
    """Strip HTML tags and collapse whitespace into clean readable text."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style blocks entirely -- they're never useful content.
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    # Collapse runs of whitespace/newlines into single spaces.
    return " ".join(text.split())


def _decode_body(payload: dict) -> str:
    """Walk a Gmail payload, preferring plain text but falling back to
    cleaned HTML so we never feed raw tags into the pipeline."""
    plain = None
    html = None

    def walk(part):
        nonlocal plain, html
        mime = part.get("mimeType", "")
        data = part.get("body", {}).get("data")
        if data:
            decoded = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
            if mime == "text/plain" and plain is None:
                plain = decoded
            elif mime == "text/html" and html is None:
                html = decoded
        for sub in part.get("parts", []):
            walk(sub)

    walk(payload)

    if plain and plain.strip():
        return plain
    if html and html.strip():
        return _html_to_text(html)
    return ""


def _header(headers: list, name: str) -> str:
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def fetch_recent_emails(max_results: int = 50) -> list:
    creds = _get_credentials()
    service = build("gmail", "v1", credentials=creds)

    results = (
        service.users()
        .messages()
        .list(userId="me", maxResults=max_results, labelIds=["INBOX"])
        .execute()
    )
    message_refs = results.get("messages", [])

    emails = []
    for ref in message_refs:
        msg = service.users().messages().get(userId="me", id=ref["id"], format="full").execute()
        payload = msg.get("payload", {})
        headers = payload.get("headers", [])
        body = _decode_body(payload) or msg.get("snippet", "")

        emails.append(
            {
                "id": msg["id"],
                "sender": _header(headers, "From") or "(unknown sender)",
                "subject": _header(headers, "Subject") or "(no subject)",
                "date": _header(headers, "Date") or "(unknown date)",
                "body": body,
            }
        )
    return emails
