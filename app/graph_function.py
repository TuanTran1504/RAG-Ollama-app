import operator
from langchain_community.document_loaders import JSONLoader
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict
from typing import List, Annotated
from langchain.schema import Document
from langgraph.graph import END
import json
from langchain_community.retrievers import BM25Retriever
import faiss
from uuid import uuid4
from langchain_community.docstore.in_memory import InMemoryDocstore
import torch
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
import getpass
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()  

tavily_key = os.getenv("TAVILY_KEY")
os.environ["TAVILY_API_KEY"] = tavily_key
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
web_search_tool = TavilySearch(k=2)
llm_json_mode = ChatOllama(model="llama3:latest", temperature=0, format="json", base_url="http://ollama:11434")
llm = ChatOllama(model="llama3:latest", temperature=0, base_url="http://ollama:11434")
llm_generate = ChatOllama(model="llama3:latest", temperature=0, base_url="http://ollama:11434")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    question: str  # User question
    generation: str  # LLM generation
    web_search: str  # Binary decision to run web search
    max_retries: int  # Max number of retries for answer generation
    answers: int  # Number of answers generated
    loop_step: Annotated[int, operator.add]
    documents: List[str]  # List of retrieved documents


### Nodes
def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    path = "./tasmania_legislation"
# Initialize JSONLoader and load JSON documents
    data = []  # This will store the data from all files

        # Iterate over all files in the folder
    for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):  # Ensure it's a file, not a subfolder
                    loader = JSONLoader(
                            file_path=file_path,
                            jq_schema=".text",  # Adjust this to your schema if needed
                            text_content=False,
                        )
                        
                    file_data = loader.load()
                    for d in file_data:
                        data.append(d)
                        
    print(f"Loaded data from {len(data)} files.")

 
    embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda"}#Moves model to GPU
        )
    index = faiss.IndexFlatL2(len(embedding_model.embed_query("hello world")))
    vector_store = FAISS(index=index, embedding_function=embedding_model, docstore= InMemoryDocstore(), index_to_docstore_id={})
    uuids = [str(uuid4()) for _ in range(len(data))]
    vector_store.add_documents(documents=data, ids=uuids)
    retriever = vector_store.as_retriever(search_type="mmr",search_kwargs={"k":2})
    bm25_retriever = BM25Retriever.from_documents(data)
    bm25_retriever.k = 2
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever], weights=[0.5,0.5])
    documents = ensemble_retriever.invoke(question)
 
    return {"documents": documents}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)
    rag_prompt = """You are a legal advisor and an assistant for question-answering tasks related to any legislation concern
    
    Answer the query in a clear, accurate, and concise manner.
    Here is the context to use to answer the question:

    {context} 

    Think carefully about the above context. 

    Now, review the user question:

    {question}

    Provide an answer to this questions using only the above context
    
    Responses should be short and direct, delivering the necessary information without filler.
    If the question is broad, please provide a more general answer

    Answer:"""


    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation, "loop_step": loop_step + 1}
def generate_normal(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    loop_step = state.get("loop_step", 0)

    generation = llm.invoke(question)
    return {"generation": generation, "loop_step": loop_step + 1, "documents":None}

def route_question(state):
    """
    Route question to web search or RAG

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    router_instructions = """You are an expert at routing a user question to a vectorstore or not.

    The vectorstore contains documents related to Tasmanian Legislation.


    Return JSON with single key, datasource, that is 'vectorstore' or 'normal' depending on the query if it needs the aditional knowledge from the vectorstore or not"""

    route_question = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=state["question"])]
    )
    source = json.loads(route_question.content)["datasource"]
    if source == "vectorstore":
        return "retrieve"
    elif source == "normal":
        return "generate_normal"
    
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    # Score each doc
    filtered_docs = []
    found_relevant_document = 0
    web_search = "No"
    doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.
    

    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

    # Grader prompt
    doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

    This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

    Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=d.page_content, question=question
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]

        # Document is relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
            found_relevant_document += 1  # Mark that we've found a relevant document.
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # Ignore the document and continue checking others.

        # Only set web_search to "Yes" if no relevant document was found
    web_search = "Yes" if found_relevant_document == 0 else "No"

    # Return filtered documents and web search state based on relevance check
    return {"documents": filtered_docs, "web_search": web_search}

def web_search(state):
    """
    Web search based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    # Web search
    docs = web_search_tool.invoke({"query": question})
    if isinstance(docs, dict) and "results" in docs:
        docs = docs["results"]  # Tavily format
    elif isinstance(docs, str):
        docs = [{"content": docs}]
    elif isinstance(docs, dict):
        docs = [docs]

    # 🧼 Step 3: Extract content only
    new_documents = [
        Document(page_content=d["content"], metadata={"source": d.get("url", "unknown")})
        for d in docs if "content" in d
    ]

    # Append to existing documents
    documents.extend(new_documents)


    return {"documents": documents}


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

