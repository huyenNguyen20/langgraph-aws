import { config } from "dotenv";
config();

import * as tslab from "tslab";
import { tool } from '@langchain/core/tools';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { BaseMessage, AIMessage, HumanMessage } from "@langchain/core/messages";
import { z } from 'zod';
import { BedrockChat } from "@langchain/community/chat_models/bedrock";
import { END, MessagesAnnotation, START, StateGraph } from "@langchain/langgraph";

// 1. Create tools
const getWeather = tool((input) => {
    if (['sf', 'san francisco'].includes(input.location.toLowerCase())) {
        return 'It\'s 60 degrees and foggy.';
    } else {
        return 'It\'s 90 degrees and sunny.';
    }
}, {
    name: 'get_weather',
    description: 'Call to get the current weather.',
    schema: z.object({
        location: z.string().describe("Location to get the weather for."),
    })
})

const getCoolestCities = tool(() => {
    return 'nyc, sf';
}, {
    name: 'get_coolest_cities',
    description: 'Get a list of coolest cities',
    schema: z.object({
        noOp: z.string().optional().describe("No-op parameter."),
    })
})

const tools = [getWeather, getCoolestCities]

// 2. Initial chat model and bind it to tools
const modelWithTools = new BedrockChat({
    model: process.env.LLM_MODEL_ID,
    region: process.env.AWS_REGION,
    temperature: 0
}).bindTools(tools);

// 3. Create tool node 
const toolNodeForGraph = new ToolNode(tools)


// 4. Create condition function
const shouldContinue = (state: typeof MessagesAnnotation.State) => {
    const { messages } = state;
    const lastMessage = messages[messages.length - 1];
    if ("tool_calls" in lastMessage && Array.isArray(lastMessage.tool_calls) && lastMessage.tool_calls?.length) {
        return "tools";
    }
    return END;
}

const callModel = async (state: typeof MessagesAnnotation.State) => {
    const { messages } = state;
    const response = await modelWithTools.invoke(messages);
    return { messages: response };
}

// 5. Create the graph
const workflow = new StateGraph(MessagesAnnotation)
    // Define the two nodes we will cycle between
    .addNode("agent", callModel)
    .addNode("tools", toolNodeForGraph)
    .addEdge(START, "agent")
    .addConditionalEdges("agent", shouldContinue, ["tools", END])
    .addEdge("tools", "agent");

const app = workflow.compile()

// const drawableGraph = app.getGraph();
// const image = await drawableGraph.drawMermaidPng();
// const arrayBuffer = await image.arrayBuffer();

// await tslab.display.png(new Uint8Array(arrayBuffer));

// example with a single tool call
const stream = await app.stream(
    {
        messages: [{ role: "user", content: "what's the weather in sf?" }],
    },
    {
        streamMode: "values"
    }
)
for await (const chunk of stream) {
    const lastMessage = chunk.messages[chunk.messages.length - 1];
    const type = lastMessage._getType();
    const content = lastMessage.content;
    const toolCalls = lastMessage.tool_calls;
    console.dir({
        type,
        content,
        toolCalls
    }, { depth: null });
}

// example with a multiple tool calls in succession
const streamWithMultiToolCalls = await app.stream(
    {
        messages: [{ role: "user", content: "what's the weather in the coolest cities?" }],
    },
    {
        streamMode: "values"
    }
)
for await (const chunk of streamWithMultiToolCalls) {
    const lastMessage = chunk.messages[chunk.messages.length - 1];
    const type = lastMessage._getType();
    const content = lastMessage.content;
    const toolCalls = lastMessage.tool_calls;
    console.dir({
        type,
        content,
        toolCalls
    }, { depth: null });
}