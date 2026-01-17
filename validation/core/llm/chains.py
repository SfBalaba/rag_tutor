from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from core.config import config
from core.llm import get_llm
from core.ranking import get_doc_content, rerank_documents
from core.vector_store import get_vector_store


def format_docs(docs):
    return config["docs_separator"].join(get_doc_content(doc) for doc in docs)


def get_retrieval_chain(
    query_transformer=None,
    top_search=config["database"]["search_top_k"],
    top_rerank=config["reranker"]["top_k"],
    use_reranker=config["reranker"]["enabled"],
):
    qa_prompt = config["qa_prompt"]
    if "qwen" in config["model"]["name"].lower():
        qa_prompt = "/no_think\n\n" + qa_prompt

    prompt = ChatPromptTemplate.from_template(qa_prompt)
    llm = get_llm()

    def retrieve_and_process(query):
        transformed_query = query_transformer(query) if query_transformer else query
        docs = get_vector_store().similarity_search(transformed_query, k=top_search)
        if use_reranker:
            docs = rerank_documents(query, docs)
        return format_docs(docs[:top_rerank])

    rag_chain = (
        {"context": retrieve_and_process, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
