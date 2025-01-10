from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import snapshot_download
import os

# Step 1: Check if Data Exists
persist_directory = "./chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

if os.path.exists(persist_directory):
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
else:
    loader = DirectoryLoader(
        '../beercss/docs/',
        glob='**/*.md',
        show_progress=True
    )
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    vector_store = Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory=persist_directory
    )

# Specify the model repository and download directory
repo_id = "unsloth/Llama-3.3-70B-Instruct-GGUF"
local_model_dir = "./models/llama-3.3"

# Download the model
snapshot_download(repo_id=repo_id, local_dir=local_model_dir)

print(f"Llama 3.3 model downloaded to {local_model_dir}")


# Step 2: Load Llama Model
llm = LlamaCpp(
    model_path="/path/to/llama-3.3.gguf",
    temperature=0.7,
    max_tokens=512,
    n_ctx=2048,
    n_gpu_layers=30,
    verbose=True
)

# Step 3:
# Step 6: Create Retrieval-Based QA Chain
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
custom_prompt = PromptTemplate(
    template="""
    You are an expert in Beer CSS.
    Answer the following question based on the provided context:

    Context: {context}
    Question: {question}

    Provide a clear and detailed response that includes explanations and examples if necessary.
    """,
    input_variables=["context", "question"]
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# Step 7: Query the Model
query = "How do I create a responsive button in Beer CSS?"
result = qa_chain.run(query)

print("Answer:", result['result'])

# Debugging Sources
for doc in result['source_documents']:
    print("Source:", doc.metadata["source"])

