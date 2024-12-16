# langgraph-aws

## What
- `01_simple_graph_agent.ts`: show how to set up a simple graph agent in langGraph, using Bedrock Foundation model
- `02_agentic_rag.ts`: show how to set up a RAG agent. The agent can not only retrieve similar information from the vector store, but also can rewrite and re-polish the human's message to create the best possible prompt. From the polished prompt, the comprehensive information can be retrieved from the vector store. The demo uses Bedrock Foundation Model and Pinecone vector store
- `03_multi_agent.ts`: show how to coordinate different agent to achieve the user's goal. In this case, the researcher agent, who searches for information in the web, coordinates with the graph agent, who creates graph from information scraped from the web, to create a graph from the collected data and send back to user
- `04_tool_calling.ts`: show how to call tool in Langgraph agentic chatbot. The foundation model can call different tools in its toolbox to fullfil the user's requirements. In this case, the agent makes api calls to 2 different APIs to get back the coolest city list and the weather in the city.
- `05_persistence.ts`: show how to create the memory persistence in langGraph chatbot.
- `06_human_in_the_loop.ts`: show how to pause the graph's execution and wait for human's approval before continuing.

## How to run
- run `npm install`
- populate the .env file
- run `npx tsx src/{filename}` e.g. npx tsx src/01_simple_graph_agent.ts

