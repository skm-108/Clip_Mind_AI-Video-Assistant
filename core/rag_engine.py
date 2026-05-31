import os
import re
from pathlib import Path
from dotenv import load_dotenv
from core.vector_store import build_vector_store, load_vector_store, get_retriever

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)

def get_llm():
    try:
        from langchain_mistralai import ChatMistralAI
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "langchain-mistralai is required for RAG chat. Install dependencies with 'pip install -r requirements.txt'."
        ) from exc

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set in .env")
    return ChatMistralAI(
        model_name="mistral-small-latest",
        api_key=api_key,
        temperature=0.3,
    )

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def _fallback_answer(question: str, docs) -> str:
    context = " ".join(doc.page_content for doc in docs[:4])
    if not context.strip():
        return "I could not find this information in the meeting transcript."

    q_terms = [term for term in re.findall(r"[a-zA-Z0-9]+", question.lower()) if len(term) > 2]
    if q_terms:
        sentences = re.split(r'(?<=[.!?])\s+', context)
        scored = []
        for sentence in sentences:
            lowered = sentence.lower()
            score = sum(1 for term in q_terms if term in lowered)
            if "?" in sentence:
                score -= 1
            if score:
                scored.append((score, sentence.strip()))
        if scored:
            scored.sort(key=lambda item: (-item[0], len(item[1])))
            answer = scored[0][1]
            return answer if answer else "I could not find this information in the meeting transcript."

    return context[:500] if context else "I could not find this information in the meeting transcript."


class _FallbackRAGChain:
    def __init__(self, retriever):
        self._retriever = retriever

    def invoke(self, question: str) -> str:
        docs = self._retriever.invoke(question)
        return _fallback_answer(question, docs)


class _SafeRAGChain:
    def __init__(self, rag_chain, retriever):
        self._rag_chain = rag_chain
        self._retriever = retriever

    def invoke(self, question: str) -> str:
        try:
            return self._rag_chain.invoke(question)
        except Exception:
            docs = self._retriever.invoke(question)
            return _fallback_answer(question, docs)

def build_rag_chain(transcript:str):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda

    vector_store = build_vector_store(transcript)

    retriever = get_retriever(vector_store, k = 4)

    try:
        llm = get_llm()
    except Exception:
        return _FallbackRAGChain(retriever)

    prompt = ChatPromptTemplate.from_messages(

        [(
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ]
    )

    #full LCEL Rag pipeline 

    rag_chain = (

        {"context" : retriever | RunnableLambda(format_docs),
         "question": RunnablePassthrough()
         }
         |prompt|llm|StrOutputParser()
    )

    return _SafeRAGChain(rag_chain, retriever)


def load_rag_chain():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda

    vector_store = load_vector_store()
    retriever = get_retriever(vector_store)

    try:
        llm = get_llm()
    except Exception:
        return _FallbackRAGChain(retriever)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context":  retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return _SafeRAGChain(rag_chain, retriever)


def ask_question(rag_chain, question:str) -> str:
    print(f"Question : {question}")
    try:
        answer = rag_chain.invoke(question)
    except Exception:
        docs = []
        retriever = getattr(rag_chain, "_retriever", None)
        if retriever is not None:
            try:
                docs = retriever.invoke(question)
            except Exception:
                docs = []
        answer = _fallback_answer(question, docs)
    print(f"answer :{answer}")
    return answer
