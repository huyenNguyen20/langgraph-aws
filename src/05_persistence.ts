import env from "dotenv";
env.config();

import { Annotation, END, MemorySaver, START, StateGraph } from "@langchain/langgraph";
import { AIMessage, BaseMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { BedrockChat } from "@langchain/community/chat_models/bedrock";
import { RunnableConfig } from "@langchain/core/runnables";
import { BufferMemory } from "langchain/memory";
import { DynamoDBChatMessageHistory } from "@langchain/community/stores/message/dynamodb";

const GraphState = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
        reducer: (x, y) => x.concat(y),
    }),
});

// 1. --- Define the tool --- 
const searchTool = tool(async ({ }: { query: string }) => {
    // This is a placeholder for the actual implementation
    return "Cold, with a low of 13 ℃";
}, {
    name: "search",
    description:
        "Use to surf the web, fetch current information, check the weather, and retrieve other information.",
    schema: z.object({
        query: z.string().describe("The query to use in your search."),
    }),
});

await searchTool.invoke({ query: "What's the weather like?" });

const tools = [searchTool];

const toolNode = new ToolNode(tools);

// 2. --- Set up Model ---
// Set up the model
const boundModel = new BedrockChat({
    model: process.env.LLM_MODEL_ID,
    region: process.env.AWS_REGION,
    temperature: 0
}).bindTools(tools);

// 3. --- Define the graph ---
const routeMessage = (state: typeof GraphState.State) => {
    const { messages } = state;
    const lastMessage = messages[messages.length - 1] as AIMessage;
    // If no tools are called, we can finish (respond to the user)
    if (!lastMessage.tool_calls?.length) {
        return END;
    }
    // Otherwise if there is, we continue and call the tools
    return "tools";
};

const callModel = async (
    state: typeof GraphState.State,
    config?: RunnableConfig,
) => {
    const { messages } = state;
    const response = await boundModel.invoke(messages, config);
    return { messages: [response] };
};

const workflow = new StateGraph(GraphState)
    .addNode("agent", callModel)
    .addNode("tools", toolNode)
    .addEdge(START, "agent")
    .addConditionalEdges("agent", routeMessage)
    .addEdge("tools", "agent");

const graph = workflow.compile();

// 4. --- Run the graph ---
// let inputs = { messages: [{ role: "user", content: "Hi I'm Yu, nice to meet you." }] };
// for await (
//     const { messages } of await graph.stream(inputs, {
//         streamMode: "values",
//     })
// ) {
//     let msg = messages[messages?.length - 1];
//     if (msg?.content) {
//         console.log(msg.content);
//     } else if (msg?.tool_calls?.length > 0) {
//         console.log(msg.tool_calls);
//     } else {
//         console.log(msg);
//     }
//     console.log("-----\n");
// }

// inputs = { messages: [{ role: "user", content: "Remember my name?" }] };
// for await (
//     const { messages } of await graph.stream(inputs, {
//         streamMode: "values",
//     })
// ) {
//     let msg = messages[messages?.length - 1];
//     if (msg?.content) {
//         console.log(msg.content);
//     } else if (msg?.tool_calls?.length > 0) {
//         console.log(msg.tool_calls);
//     } else {
//         console.log(msg);
//     }
//     console.log("-----\n");
// }

// 5. --- create conversational memory --- 
// const memory = new MemorySaver();
const memory = new BufferMemory({
    chatHistory: new DynamoDBChatMessageHistory({
        tableName: process.env.AWS_DYNAMODB_TABLE_NAME!,
        partitionKey: "id",
        sessionId: new Date().toISOString(), // Or some other unique identifier for the conversation
        config: {
            region: process.env.AWS_REGION
        },
    }),
});
const persistentGraph = workflow.compile({ checkpointer: memory });

let config = { configurable: { thread_id: "conversation-num-1" } };
let inputs = { messages: [{ role: "user", content: "Hi I'm Jo, nice to meet you." }] };
for await (
    const { messages } of await persistentGraph.stream(inputs, {
        ...config,
        streamMode: "values",
    })
) {
    let msg = messages[messages?.length - 1];
    if (msg?.content) {
        console.log(msg.content);
    } else if (msg?.tool_calls?.length > 0) {
        console.log(msg.tool_calls);
    } else {
        console.log(msg);
    }
    console.log("-----\n");
}

inputs = { messages: [{ role: "user", content: "Remember my name?" }] };
for await (
    const { messages } of await persistentGraph.stream(inputs, {
        ...config,
        streamMode: "values",
    })
) {
    let msg = messages[messages?.length - 1];
    if (msg?.content) {
        console.log(msg.content);
    } else if (msg?.tool_calls?.length > 0) {
        console.log(msg.tool_calls);
    } else {
        console.log(msg);
    }
    console.log("-----\n");
}