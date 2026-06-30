"""
Turns raw email dicts into chunked LangChain Documents ready to embed.
Each chunk carries metadata including the email's category tag, which
gets stored in Chroma and surfaced in the frontend UI.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=300,
)

CATEGORIES = ["Urgent", "Finance", "Work", "Personal", "Promotional", "Travel", "Social", "Other"]


def categorize_email(subject: str, body_preview: str) -> str:
    """Ask Claude Haiku to classify the email into one of the fixed categories."""
    import anthropic
    from app.config import settings

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    prompt = (
        f"Classify this email into exactly one of these categories: {', '.join(CATEGORIES)}.\n"
        f"Reply with only the category name, nothing else.\n\n"
        f"Subject: {subject}\n"
        f"Preview: {body_preview[:300]}"
    )
    message = client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )
    label = message.content[0].text.strip()
    return label if label in CATEGORIES else "Other"


def emails_to_documents(emails: list) -> list:
    documents = []
    for email in emails:
        category = categorize_email(email["subject"], email.get("body", ""))

        full_text = (
            f"From: {email['sender']}\n"
            f"Subject: {email['subject']}\n"
            f"Date: {email['date']}\n"
            f"Category: {category}\n\n"
            f"{email['body']}"
        )
        chunks = _splitter.split_text(full_text)
        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "email_id": email.get("id", f"unknown-{i}"),
                        "sender": email["sender"],
                        "subject": email["subject"],
                        "date": email["date"],
                        "category": category,
                        "chunk_index": i,
                    },
                )
            )
    return documents