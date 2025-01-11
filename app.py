import os
from typing import List

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
)
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load the .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError(
        "Environment variable 'OPENAI_API_KEY' is not set. Please check your .env file or environment settings."
    )

# Specify the directory containing your Markdown files
DOCS_DIRECTORY = "../beercss/docs/"
HUGGINGFACE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
EMBEDDER_MODEL_NAME = "all-mpnet-base-v2"
MAX_NEW_TOKENS = 512


class E5Embedder:
    def __init__(self):
        self.model = SentenceTransformer("intfloat/e5-large-v2")
        print("E5 model initialized successfully!")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # E5 often recommends prepending "passage: " or "query: "
        # for best performance. We'll do "passage: {text}" for documents.
        inputs = [f"passage: {t}" for t in texts]
        return self.model.encode(inputs, show_progress_bar=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([f"query: {text}"], show_progress_bar=False)[0]


def load_multi_gpu_model(model_name: str) -> HuggingFacePipeline:
    tokenizer_obj = AutoTokenizer.from_pretrained(model_name)
    model_obj = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model_obj,
        tokenizer=tokenizer_obj,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    return HuggingFacePipeline(pipeline=text_gen_pipeline)


class MarkdownPreprocessorPlain:
    def __init__(self, filepath):
        self.filepath = filepath

    def lazy_load(self):
        with open(self.filepath, "r", encoding="utf-8") as file:
            content = file.read()

        yield Document(page_content=content, metadata={"source": self.filepath})


class MarkdownEmbedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDER_MODEL_NAME)
        print("Embedder model initialized successfully!")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embeddings for a single query.
        """
        return self.model.encode([text]).tolist()[0]


def retriever(is_open_ai: bool):
    # Initialize the DirectoryLoader with UnstructuredMarkdownLoader for .md files
    loader = DirectoryLoader(
        DOCS_DIRECTORY, glob="**/*.md", loader_cls=MarkdownPreprocessorPlain
    )

    # Load the documents
    documents = loader.load()

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=300, separators=["\n\n", "\n", " "]
    )

    # Split the documents into chunks
    document_chunks = text_splitter.split_documents(documents)

    # Initialize the OpenAI embeddings
    if is_open_ai:
        embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    else:
        embedding_model = E5Embedder()

    # Store the embeddings in ChromaDB
    vectorstore = Chroma.from_documents(document_chunks, embedding_model)

    # Set up the retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever


def qa_chain(is_open_ai: bool):
    # Initialize the chat model
    if is_open_ai:
        llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY)
    else:
        llm = load_multi_gpu_model(HUGGINGFACE_MODEL_NAME)

    # Define a prompt template for the QA chain
    prompt_template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
    Provide a detailed and comprehensive explanation, including any examples or code snippets.

    {context}

    Question: {question}
    Answer:"""

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever(is_open_ai),
        chain_type="stuff",  # or "map_reduce", "refine" based on your needs
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


def answer_query(query, is_open_ai: bool):
    # Use the invoke method to run the chain
    response = qa_chain(is_open_ai).invoke({"query": query})
    return response["result"]  # Extract the result key from the response


def main():
    # Example usage
    query = "How do I create a responsive button in Beer CSS?"
    is_open_ai = False
    answer = answer_query(query, is_open_ai)
    print(answer)


if __name__ == "__main__":
    main()
