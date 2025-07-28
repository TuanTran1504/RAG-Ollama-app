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

os.environ["TAVILY_API_KEY"] = "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"
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
            model_name="nlpaueb/legal-bert-base-uncased",
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
    
    Based strictly on the provided context, answer the query in a clear, accurate, and concise manner.
    Here is the context to use to answer the question:

    {context} 

    Think carefully about the above context. 

    Now, review the user question:

    {question}

    Provide an answer to this questions using only the above context
    
    Responses should be short and direct, delivering the necessary information without filler.

    Answer:"""


    #print(documents)
    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation, "loop_step": loop_step + 1}

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
    if isinstance(docs, str):
        # If it's a single string, wrap it
        docs = [{"content": docs}]
    elif isinstance(docs, dict):
        # If somehow a single dict, wrap it into a list
        docs = [docs]
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents=[web_results]
    return {"documents": documents}

def route_question(state):
    """
    Route question to web search or RAG

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    router_instructions = """You are an expert at routing a user question to a vectorstore or web search.

    The vectorstore1 contains documents related to Western Sydney University International College.

    Use the vectorstore2 for questions related to  Western Sydney University (not International College).

    Return JSON with single key, datasource, that is 'vectorstore1' or 'vectorstore2' depending on the question."""

    route_question = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=state["question"])]
    )
    source = json.loads(route_question.content)["datasource"]
    if source == "vectorstore1":
        print("---ROUTE QUESTION TO WSUIC DATABASE---")
        return "retrieve"
    elif source == "vectorstore2":
        print("---ROUTE QUESTION TO WSU DATABASE---")
        return "retrieve2"

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

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)  # Default to 3 if not provided
    hallucination_grader_instructions = """

    You are a teacher grading a quiz. 

    You will be given FACTS and a STUDENT ANSWER. 

    Here is the grade criteria to follow:

    (1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

    (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

    Score:

    A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

    A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

    Avoid simply stating the correct answer at the outset."""

    # Grader prompt
    hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

    Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""
    answer_grader_instructions = """You are a teacher grading a quiz. 

    You will be given a QUESTION and a STUDENT ANSWER. 

    Here is the grade criteria to follow:

    (1) The STUDENT ANSWER helps to answer the QUESTION

    Score:

    A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

    The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

    A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

    Avoid simply stating the correct answer at the outset."""

    # Grader prompt
    answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

    Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(documents), generation=generation.content
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)["binary_score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        # Test using question and generation from above
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, generation=generation.content
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"
    elif state["loop_step"] <= max_retries:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    else:
        print("---DECISION: MAX RETRIES REACHED---")
        return "max retries"