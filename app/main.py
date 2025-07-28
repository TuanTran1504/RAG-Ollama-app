import asyncio
from langgraph.graph import StateGraph
from graph_function import *
from langgraph.graph import END, START
import os
import json
import pandas as pd
from langchain_ollama import ChatOllama
# from datasets import Dataset
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.smith import RunEvalConfig


# Define the workflow
workflow = StateGraph(GraphState)
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("generate_normal", generate_normal)
workflow.add_node("route_question", route_question)

workflow.set_conditional_entry_point(
    route_question,
    {
        "retrieve":"retrieve",
        "generate_normal":"generate_normal",
    }
)
workflow.add_edge("generate_normal", END)
workflow.add_edge("websearch", "generate")
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
        if documents_retrieved:
            documents = [doc.page_content for doc in documents_retrieved if hasattr(doc, 'page_content')]

            # Append the generated result to test_schema
            test_schema.append({
                "question": question,
                "answer": generation,
                "contexts": documents
            })

            return test_schema
        else:
            test_schema.append({
                "question": question,
                "answer":generation
            })
            return test_schema


