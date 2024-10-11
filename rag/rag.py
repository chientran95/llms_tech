from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv('../.env')


llm = ChatOpenAI(model="gpt-4o-mini")

def run_llm_with_rag(question):
    vector_store = FAISS.load_local(
        "data/prepared/vectorstore/faiss_index",
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True
    )
    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question)
    print('Result with RAG: ', answer)

def run_llm_without_rag(question):
    normal_chain = (
        llm | StrOutputParser()
    )

    answer = normal_chain.invoke(question)
    print('Result without RAG: ', answer)

def main():
    question = 'What is Task Decomposition?'
    print('Question: ', question)
    run_llm_without_rag(question)
    print('\n\n\n')
    run_llm_with_rag(question)

if __name__ == '__main__':
    main()
