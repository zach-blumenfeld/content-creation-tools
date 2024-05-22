# Marketing Assistant
Assistant to help draft & edit market content using a knowledge graph of core marketing materials including MPF docs and hero demo transcripts. 


To start the project, run the following command:

```
docker-compose up
```

Open `http://localhost:8501` in your browser to interact with the assistant.

## Environment Setup

You need to define the following environment variables in the `.env` file.

```
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
NEO4J_URI=<YOUR_NEO4J_URI>
NEO4J_USERNAME=<YOUR_NEO4J_USERNAME>
NEO4J_PASSWORD=<YOUR_NEO4J_PASSWORD>
```

## Docker containers

This project contains the following services wrapped as docker containers

1**API**:
   - Uses LangChain to retrieve messaging from Neo4j and call OpenAI LLM.
2**UI**:
   - Simple streamlit chat user interface. Available on `localhost:8501`.

## Populating Database

Use the [LLM Graph Builder](https://neo4j.com/labs/genai-ecosystem/llm-graph-builder/) to populate a Neo4j database from source marketing documents

## Contributions

Contributions are welcome!