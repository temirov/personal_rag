import os
import spacy
import json
import numpy as np
import glob
import time

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)

# For clustering
from sklearn.cluster import KMeans
from typing import List

class MyDirectoryLoader:
    """
    A simple custom loader that:
      1. Recursively loads .md or .html files in a given directory.
      2. Attaches file name, file path, and modification timestamp to each Document's metadata.
    """
    def __init__(self, path: str, glob: str = "**/*.md"):
        self.directory = path
        self.glob_pattern = glob

    def load(self) -> List[Document]:
        doc_list = []
        full_pattern = os.path.join(self.directory, self.glob_pattern)
        files_found = glob.glob(full_pattern, recursive=True)

        for file_path in files_found:
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read()

            # Gather some metadata
            file_name = os.path.basename(file_path)
            mod_time = os.path.getmtime(file_path)
            mod_time_str = time.ctime(mod_time)  # or store as numeric timestamp

            metadata = {
                "source": file_name,
                "absolute_path": file_path,
                "modified_time": mod_time_str
                # You could add more (URL, images, etc.) if relevant
            }

            # Create the Document with the text content and attached metadata
            doc = Document(page_content=text_content, metadata=metadata)
            doc_list.append(doc)

        return doc_list

def extract_entities_and_relationships(
        text_value: str,
        nlp_pipeline: spacy.Language
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    doc_object = nlp_pipeline(text_value)
    entities = [(entity.text, entity.label_) for entity in doc_object.ents]
    relationships = []
    for token in doc_object:
        if token.dep_ in ("nsubj", "dobj"):
            relationships.append((token.head.text, token.text, token.dep_))
    return entities, relationships


def build_or_load_vectorstore(
        docs_directory: str,
        embeddings_model_name: str,
        persist_directory: str,
        nlp_pipeline: spacy.Language,
        use_knowledge_graph: bool = False,
        num_clusters: int = 3
) -> Chroma:
    """
    Clusters the chunks into `num_clusters` groups and keeps only the
    single 'representative' chunk from each cluster.
    """
    embedding_model = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    # Toggle as needed if you want to forcibly recreate the store
    if os.path.exists(persist_directory) and False:
        print("Loading existing Chroma vector store...")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
    else:
        print("Creating new Chroma vector store...")
        loader = MyDirectoryLoader(
            path=docs_directory,
            glob='**/*.md',
        )
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )
        chunked_documents = text_splitter.split_documents(documents)

        # 1. Create an in-memory list of chunk data
        #    (including text, metadata, and an embedding)
        chunk_data_list = []
        for chunk_object in chunked_documents:
            if use_knowledge_graph:
                entities, relationships = extract_entities_and_relationships(
                    chunk_object.page_content, nlp_pipeline
                )
                metadata_dict = {
                    "entities": json.dumps(entities),
                    "relationships": json.dumps([
                        {"source": rel_item[0], "target": rel_item[1], "type": rel_item[2]}
                        for rel_item in relationships
                    ])
                }
            else:
                metadata_dict = {}

            # We'll store the raw text now; we'll embed it in one pass below.
            chunk_data_list.append({
                "text": chunk_object.page_content,
                "metadata": metadata_dict
            })

        # 2. Embed each chunk using your HuggingFaceEmbeddings
        all_texts = [item["text"] for item in chunk_data_list]
        embeddings_list = [embedding_model.embed_query(txt) for txt in all_texts]

        # 3. Perform k-means clustering on the chunk embeddings
        print(f"Clustering {len(all_texts)} chunks into {num_clusters} clusters...")
        kmeans_cluster = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans_cluster.fit_predict(embeddings_list)

        # 4. For each cluster, pick the single 'closest' chunk to the centroid
        #    This drastically reduces repetition for repeated boilerplate text.
        cluster_to_representative_idx = {}
        centroids = kmeans_cluster.cluster_centers_

        for idx, label_value in enumerate(cluster_labels):
            # Compute distance to cluster centroid
            chunk_vector = np.array(embeddings_list[idx])
            centroid = centroids[label_value]
            distance = np.linalg.norm(chunk_vector - centroid)

            # If no rep is chosen yet, or we found a closer chunk, update
            if label_value not in cluster_to_representative_idx:
                cluster_to_representative_idx[label_value] = (idx, distance)
            else:
                current_best_idx, current_best_dist = cluster_to_representative_idx[label_value]
                if distance < current_best_dist:
                    cluster_to_representative_idx[label_value] = (idx, distance)

        # 5. Build final Document objects from only the representative chunk of each cluster
        final_documents = []
        for label_value, (best_idx, best_distance) in cluster_to_representative_idx.items():
            chunk_data = chunk_data_list[best_idx]
            cleaned_text = chunk_data["text"]
            document_obj = Document(
                page_content=cleaned_text,
                metadata=chunk_data["metadata"]
            )
            final_documents.append(document_obj)

        print(f"Kept only {len(final_documents)} representative chunks out of {len(chunked_documents)} total.")

        # 6. Create Chroma store
        return Chroma.from_documents(
            documents=final_documents,
            embedding=embedding_model,
            persist_directory=persist_directory
        )


def load_multi_gpu_model(model_name: str) -> HuggingFacePipeline:
    tokenizer_obj = AutoTokenizer.from_pretrained(model_name)
    model_obj = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model_obj,
        tokenizer=tokenizer_obj,
    )
    return HuggingFacePipeline(pipeline=text_gen_pipeline)


def create_retrieval_qa_chain(
        llm_model: HuggingFacePipeline,
        vector_store: Chroma
) -> RetrievalQA:
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,  # how many to initially consider
            "lambda_mult": 0.5
        }
    )
    custom_prompt = PromptTemplate(
        template="""
    You are an expert in Beer CSS. Based on the context provided below, answer the question clearly and concisely.

    Context:
    {context}

    Question:
    {question}

    Summarize the relevant parts and provide code samples if applicable. Then directly answer the user question.
    """,
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm_model,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt}
    )


def main():
    persist_directory = "./chroma_db"
    embedding_model_name = "BAAI/bge-large-en-v1.5"
    docs_directory = "../beercss/docs/"

    huggingface_model_name = "meta-llama/Llama-3.1-8B-Instruct"  # for example

    import en_core_web_sm
    nlp_pipeline = en_core_web_sm.load()

    # Build or load the vector store with "representative chunk" logic
    vector_store = build_or_load_vectorstore(
        docs_directory=docs_directory,
        embeddings_model_name=embedding_model_name,
        persist_directory=persist_directory,
        nlp_pipeline=nlp_pipeline,
        use_knowledge_graph=True,  # set True if desired
        num_clusters=8,  # tune as needed
    )

    # Load multi-GPU model
    hf_llm_model = load_multi_gpu_model(huggingface_model_name)

    # Create QA chain
    qa_chain = create_retrieval_qa_chain(llm_model=hf_llm_model, vector_store=vector_store)

    # Test query
    query = "How do I create a responsive button in Beer CSS?"
    result = qa_chain.invoke({"query": query})
    print("Answer:", result['result'])

    print("\nDebugging Sources:")
    for doc in result['source_documents']:
        print("Source:", doc.metadata.get("source", "No source metadata found"))


if __name__ == "__main__":
    main()
