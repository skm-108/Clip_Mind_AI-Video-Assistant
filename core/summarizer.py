import os 
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)

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
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200
    )

    return splitter.split_text(transcript)

def summarize(transcript : str) -> str:
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

    chunk_summaries = [map_chain.invoke({"text" : chunk}) for chunk in chunks]

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

    return combined_chain.invoke(combined)

def generate_title(transcipt : str) -> str:
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
        |StrOutputParser()
    )

    return title_chain.invoke(transcipt[:2000])




