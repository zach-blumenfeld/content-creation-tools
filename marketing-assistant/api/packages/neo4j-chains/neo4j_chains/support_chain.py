import json
import os
from collections import OrderedDict
from operator import itemgetter
from typing import List, Tuple, Any, Dict

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from neo4j_chains.condense_question_chain import condense_question
from neo4j_chains.utils import llm, format_doc, graph

template = (
    "You are an expert marketer for Neo4j, the Graph Database and Analytics company."
    "You are responsible for helping users with tasks around generating and editing marketing content. "
    "If you require follow up questions, "
    "make sure to ask the user for clarification. Make sure to include any "
    "available options that need to be clarified in the follow up questions. "
    "Answer the question based only on the below Rules and AdditionalContext. "
    "The Rules and AdditionalContext are extracted from the Messaging Framework which provides context "
    "to guide the formulation of all marketing content. "
    "The Messaging Framework will tell you about the strengths of Neo4j to highlight"
    "as well as the target audience you are writing for."
    "The Rules in particular should be respected as absolute fact, never provide answers that contradicts the rules. "
    """
    # Rules
    {rules}
    
    # AdditionalContext
    {additionalContext}
    
    # Task: 
    {question}
        
    # Answer:
    """
)

vector_top_k = 5

vector_retrieval_query = """RETURN node.text AS text, score, node {.*, text: Null, embedding: Null} AS metadata"""
vector_store = Neo4jVector.from_existing_index(
    embedding=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
    index_name="vector",
    retrieval_query=vector_retrieval_query
)


def format_docs(docs: List[Document]) -> str:
    print("///////////// DOCS /////////////////")
    print(json.dumps([format_doc(d) for d in docs]))
    print("///////////// END DOCS /////////////////")
    return json.dumps([format_doc(d) for d in docs], indent=1)


def retrieve_rules(docs: List[Document]) -> str:
    doc_chunk_ids = [doc.metadata['id'] for doc in docs]
    res = graph.query("""
    UNWIND $chunkIds AS chunkId
    MATCH(chunk {id:chunkId})-[:HAS_ENTITY]->()-[rl:!HAS_ENTITY]-{1,5}()
    UNWIND rl AS r
    WITH DISTINCT r
    MATCH (n)-[r]->(m)
    RETURN n.id + ' - ' + type(r) +  ' -> ' + m.id AS rule ORDER BY rule
    """, params={'chunkIds': doc_chunk_ids})
    print("///////////// RULES /////////////////")
    print([r['rule'] for r in res])
    print("///////////// END RULES /////////////////")
    return '\n'.join([r['rule'] for r in res])


class ChainInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(
        ..., extra={"widget": {"type": "chat", "input": "input", "output": "output"}}
    )


class Output(BaseModel):
    output: Any


prompt = ChatPromptTemplate.from_template(template)

qa_chain = (
        RunnableParallel({
            "vectorStoreResults": condense_question | vector_store.as_retriever(search_kwargs={'k': vector_top_k}),
            "question": RunnablePassthrough()})
        | RunnableParallel({
    "rules": (lambda x: x["vectorStoreResults"]) | RunnableLambda(retrieve_rules),
    "additionalContext": (lambda x: x["vectorStoreResults"]) | RunnableLambda(format_docs),
    "question": lambda x: x["question"]})
        | prompt
        | llm
        | StrOutputParser()
).with_types(input_type=ChainInput, output_type=Output)
