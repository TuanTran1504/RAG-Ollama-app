import asyncio
from langgraph.graph import StateGraph
from graph_function2 import *
from langgraph.graph import END, START
import os
import json
import pandas as pd
from langchain_ollama import ChatOllama
# from datasets import Dataset
from ragas import SingleTurnSample
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage
from ragas.metrics import LLMContextPrecisionWithoutReference
from langchain_openai import ChatOpenAI
from langchain.smith import RunEvalConfig
from ragas.integrations.langchain import EvaluatorChain

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy
from ragas import EvaluationDataset

# Define the workflow
workflow = StateGraph(GraphState)
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate

workflow.add_edge(START, "retrieve")
workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
graph = workflow.compile()

async def generate_responses(questions):
    test_schema = []

    # Generate responses for each question asynchronously
    for question in questions:
        inputs = {"question": question, "max_retries": 3}
        result = await graph.invoke(inputs) if asyncio.iscoroutinefunction(graph.invoke) else graph.invoke(inputs)
        generation = result.get("generation").content if "generation" in result else None
        documents_retrieved = result.get("documents", [])
        documents = [doc.page_content for doc in documents_retrieved if hasattr(doc, 'page_content')]

        # Append the generated result to test_schema
        test_schema.append({
            "question": question,
            "answer": generation,
            "contexts": documents
        })

    return test_schema

async def main():
    questions=["Under the Local Government Act 1993 (Tas), who has the authority to issue a council proclamation?"]
        # Step 1: Generation Phase
    test_schema = await generate_responses(questions)
    for data in test_schema:
            print(f"Question: {data['question']}")
            print(f"Answer: {data['answer']}")
            print(f"Document: {data['contexts']}")
# Run the async workflow
asyncio.run(main())
