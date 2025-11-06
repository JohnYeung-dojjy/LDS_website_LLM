"""Django + nestjs frontend is a bit complex for this demo project. This is a command-line test file to showcase backend functionality."""
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from prepare_db import CHROMA_DB_PATH_DIR, LLM_MODEL

SYSTEM_PROMPT_TEMPLATE = """
You are the Community Engagement Officer representing LDS -- Learn. Develop. Succeed. (Learning Disability Society).

Here are some relevant information from their website at https://ldsociety.ca/: {infos}

Here is the question to answer: {question}

Give a detailed and helpful answer based on the above information. If the information does not contain the answer, politely say that you don't know.
"""

def main():
    print("Setting up retriever and LLM...")
    embeddings = OllamaEmbeddings(model=LLM_MODEL)
    chroma_db = Chroma(
        persist_directory=str(CHROMA_DB_PATH_DIR),
        embedding_function=embeddings
    )
    retriever = chroma_db.as_retriever(search_kwargs={"k": 10})
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
    llm = OllamaLLM(model="deepseek-r1:8b")
    chain = prompt | llm
    while True:
        print("\n", end="")
        print("="*100)
        question = input("Ask your question (q to quit): ")
        print("\n")
        if question == "q":
            break

        infos = retriever.invoke(question)
        print([info.metadata["source"] for info in infos])
        result = chain.invoke({"infos": infos, "question": question})
        print(result)


if __name__ == "__main__":
    main()
