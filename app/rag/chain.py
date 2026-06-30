"""
The core RAG chain. This is where LangChain actually orchestrates the
retrieval -> prompt assembly -> LLM call pipeline.

The LLM provider is swappable via settings.LLM_PROVIDER:
  - "ollama": free, local, good for development (no API costs while testing)
  - "openai": hosted, used for the polished cloud demo

Swapping providers only changes get_llm() -- the retrieval and prompt
logic stay identical, which is the point of using LangChain's
abstractions instead of hand-rolling provider-specific code.
"""
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from app.config import settings
from app.rag.vectorstore import get_vectorstore

PROMPT_TEMPLATE = """You are an assistant that answers questions about the user's emails.
Use ONLY the context below to answer. If the answer isn't in the context, say you don't know
rather than guessing. Be concise and reference specific details (dates, amounts, names) when relevant.

CRITICAL FORMATTING RULES:
- Write in plain, normal sentences only.
- Do NOT use any markdown symbols anywhere in your answer: no asterisks (*), no underscores (_),
  no pound signs (#), no bullet points, no numbered lists, no bold, no italics.
- Always write monetary amounts with a dollar sign, like $21.46, never just 21.46.
- Your answer should read exactly like a normal typed sentence with regular spacing between every word.

Emails often contain promotional or secondary content (loyalty programs, add-on offers,
concessions, ads, upsells) mixed in alongside the main subject of the email. Carefully
distinguish the primary subject (e.g. the actual movie being seen, the actual product
purchased, the actual appointment scheduled) from any secondary brand names, program names,
or promotional items mentioned nearby. When identifying a title, name, or subject, prefer
the one that matches the email's main purpose (e.g. the subject line, order confirmation,
or ticket details) over a name that appears in a sidebar, ad, or perks section.

Context:
{context}

Question: {question}

Answer:"""


DRAFT_REPLY_TEMPLATE = """You are helping a user write a reply to an email they received.

Email details:
Subject: {subject}
From: {sender}
Date: {date}
Content: {snippet}

Write a {tone} reply to this email. The reply should:
- Be appropriate for the email's content and context
- Sound natural, like a human wrote it (not a form letter)
- Not include a subject line -- just the body
- End with a neutral sign-off like "Best," or "Thanks," followed by a blank line for the user's name

CRITICAL FORMATTING RULES:
- Plain sentences only. No markdown, no asterisks, no bullet points, no bold, no italics.
- Write monetary amounts with a dollar sign ($21.46), never bare numbers.

Reply:"""


def get_llm():
    if settings.LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.ANTHROPIC_API_KEY:
            raise RuntimeError("LLM_PROVIDER is 'anthropic' but ANTHROPIC_API_KEY is not set.")
        return ChatAnthropic(model=settings.ANTHROPIC_MODEL, api_key=settings.ANTHROPIC_API_KEY, temperature=0)

    if settings.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.OPENAI_API_KEY:
            raise RuntimeError("LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set.")
        return ChatOpenAI(model=settings.OPENAI_MODEL, api_key=settings.OPENAI_API_KEY, temperature=0)

    from langchain_community.llms import Ollama

    return Ollama(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_MODEL, temperature=0)


def build_qa_chain():
    vectorstore = get_vectorstore()
    llm = get_llm()
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": settings.RETRIEVAL_K}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def ask_question(question: str):
    chain = build_qa_chain()
    result = chain.invoke({"query": question})

    answer = result["result"]

    sources = []
    seen = set()
    for doc in result["source_documents"]:
        key = (doc.metadata.get("subject"), doc.metadata.get("sender"))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "subject": doc.metadata.get("subject", "Unknown"),
                "sender": doc.metadata.get("sender", "Unknown"),
                "date": doc.metadata.get("date", "Unknown"),
                "snippet": doc.page_content[:200],
                "category": doc.metadata.get("category", "Other"),
            }
        )

    return answer, sources


def draft_reply(subject: str, sender: str, date: str, snippet: str, tone: str = "professional") -> str:
    llm = get_llm()
    prompt = PromptTemplate(
        template=DRAFT_REPLY_TEMPLATE,
        input_variables=["subject", "sender", "date", "snippet", "tone"],
    )
    chain = prompt | llm
    result = chain.invoke({
        "subject": subject,
        "sender": sender,
        "date": date,
        "snippet": snippet,
        "tone": tone,
    })
    return result.content if hasattr(result, "content") else str(result)