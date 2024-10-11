import os
import sys
import bs4
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader, BSHTMLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

import faiss
from uuid import uuid4
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

load_dotenv('../.env')


llm = ChatOpenAI(model="gpt-4o-mini")

def process_data(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    splits = text_splitter.split_documents(docs)
    print('Data splited: ', len(splits), len(splits[0].page_content), splits[10].metadata)

    index = faiss.IndexFlatL2(len(OpenAIEmbeddings().embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=OpenAIEmbeddings(),
        index=index,
        docstore= InMemoryDocstore(),
        index_to_docstore_id={}
    )
    uuids = [str(uuid4()) for _ in range(len(splits))]
    vector_store.add_documents(documents=splits, ids=uuids)
    vector_store_ouput = os.path.join('data', 'prepared', 'vectorstore', "faiss_index")
    os.makedirs(os.path.join('data', 'prepared', 'vectorstore'), exist_ok=True)
    vector_store.save_local(vector_store_ouput)
    print(f'Vector store created and saved to {vector_store_ouput}')

def main():
    # Use data from webpage
    # loader = WebBaseLoader(
    #     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    #     bs_kwargs=dict(
    #         parse_only=bs4.SoupStrainer(
    #             class_=("post-content", "post-title", "post-header")
    #         )
    #     ),
    # )
    # docs = loader.load()
    # print(len(docs[0].page_content))

    file_path = sys.argv[1]
    loader = BSHTMLLoader(
        file_path,
        open_encoding='utf-8',
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    process_data(docs)

if __name__ == '__main__':
    main()