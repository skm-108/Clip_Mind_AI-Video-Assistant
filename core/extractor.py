import os 
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)


def get_llm():
    try:
        from langchain_mistralai import ChatMistralAI
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "langchain-mistralai is required for action items, decisions, and question extraction. Install dependencies with 'pip install -r requirements.txt'."
        ) from exc

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set in .env")
    return ChatMistralAI(
        model_name="mistral-small-latest",
        api_key=api_key,
        temperature=0.2,
    )



def build_chain(system_prompt : str):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda

    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )

def extract_action_items(transcript:str)->str:
    chain = build_chain(
         "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )

    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return chain.invoke(transcript)
