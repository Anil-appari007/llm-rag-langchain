from typing import AsyncIterator, Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


file_path = (
    "Commands.pdf"
)
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()
print(len(pages))
print(pages[0].page_content)


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(pages)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# embeddings = embeddings_model.embed_documents(texts)
# len(embeddings)
# print(embeddings)

vector_store = Chroma.from_documents(documents=chunks,embedding=embeddings)

# print(vector_store)

retriever = vector_store.as_retriever()
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm = ChatOpenAI()
template = """
            SYSTEM: You are a question answer bot.
            Be factual in your response.
            Respond to the following question : {question} only from the below context : {context} .
            If you don't know the answer, say you don't know
        """
prompt = PromptTemplate.from_template(template)

chain = (
    {"context":retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(chain.invoke("how to run app"))