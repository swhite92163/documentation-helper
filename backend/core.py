import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.embeddings.openai import OpenAIEmbeddings
# from langchain_community.chat_models.openai import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from pinecone import Pinecone

from constants import INDEX_NAME


load_dotenv()

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
)


def run_llm(query: str) -> any:
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeLangChain.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    return qa({"query": query})
