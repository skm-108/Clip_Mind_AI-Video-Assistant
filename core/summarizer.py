import os 
from pathlib import Path
import re
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)

_MISTRAL_DISABLED = False


def _mistral_enabled() -> bool:
    return bool(os.getenv("MISTRAL_API_KEY")) and not _MISTRAL_DISABLED


def _disable_mistral() -> None:
    global _MISTRAL_DISABLED
    _MISTRAL_DISABLED = True

def get_llm():
    try:
        from langchain_mistralai import ChatMistralAI
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "langchain-mistralai is required for summarization and title generation. Install dependencies with 'pip install -r requirements.txt'."
        ) from exc

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set in .env")
    return ChatMistralAI(
        model_name="mistral-small-latest",
        api_key=api_key,
        temperature=0.3,
    )


def split_transcript(transcript: str) -> list:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=200,
        )

        return splitter.split_text(transcript)
    except ModuleNotFoundError:
        # Fallback splitter: split by paragraphs and sentences to approximate chunks
        pieces = [p.strip() for p in re.split(r"\n{2,}", transcript) if p.strip()]
        if not pieces:
            pieces = [s.strip() for s in re.split(r'(?<=[.!?]) +', transcript) if s.strip()]

        chunks = []
        current = ""
        for piece in pieces:
            if len(current) + len(piece) + 2 <= 3000:
                current = (current + "\n\n" + piece).strip()
            else:
                if current:
                    chunks.append(current)
                if len(piece) <= 3000:
                    current = piece
                else:
                    # Hard split long piece
                    for i in range(0, len(piece), 3000 - 200):
                        chunks.append(piece[i : i + (3000 - 200)])
                    current = ""
        if current:
            chunks.append(current)
        return chunks


def _simple_summary(transcript: str, max_sentences: int = 5) -> str:
    """Lightweight fallback summarizer when Mistral is unavailable.
    Picks the longest sentences up to `max_sentences` as a naive summary.
    """
    if not transcript:
        return ""
    # Split into rough sentences.
    candidates = [s.strip() for s in re.split(r'[\n\r]+|(?<=[.!?]) +', transcript) if s.strip()]
    if not candidates:
        return transcript[:500]
    # Score by length (simple heuristic) and pick top sentences in original order.
    scored = sorted(((len(s), i, s) for i, s in enumerate(candidates)), reverse=True)
    top = sorted(scored[:max_sentences], key=lambda x: x[1])
    return "\n\n".join(s for _, _, s in top)


def _simple_title(transcript: str, max_words: int = 8) -> str:
    """Naive title generator: use the first sentence or first words as a title."""
    if not transcript:
        return "Untitled"
    first = transcript.strip().split('\n', 1)[0]
    # If the first line is long, take first sentence.
    first_sentence = re.split(r'(?<=[.!?]) +', first)[0]
    words = first_sentence.split()
    title = " ".join(words[:max_words])
    title = title.strip(' .,!?:;\n\r')
    return title or "Untitled"

def summarize(transcript : str) -> str:
    # If no Mistral API key is available, use a lightweight fallback.
    if not _mistral_enabled():
        return _simple_summary(transcript)

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda

    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize this portion of a meeting transcript concisely."),
            ("human", "{text}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    try:
        chunk_summaries = [map_chain.invoke({"text" : chunk}) for chunk in chunks]
    except Exception:
        _disable_mistral()
        return _simple_summary(transcript)

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting summarizer. Combine these partial summaries "
                "into one final professional meeting summary in bullet points.",
            ),
            ("human", "{text}"),
        ]
    )

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | combined_prompt | llm | StrOutputParser()
    )

    try:
        return combined_chain.invoke(combined)
    except Exception:
        _disable_mistral()
        return _simple_summary(transcript)

def generate_title(transcipt : str) -> str:
    # If Mistral API key is not set, return a simple heuristic title.
    if not _mistral_enabled():
        return _simple_title(transcipt)

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda

    llm = get_llm()

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | 
        ChatPromptTemplate.from_messages([
            (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )

    try:
        return title_chain.invoke(transcipt[:2000])
    except Exception:
        _disable_mistral()
        return _simple_title(transcipt)




