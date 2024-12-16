import env from "dotenv";
env.config();

import { StateGraph, START, END, Annotation } from "@langchain/langgraph";
import { MemorySaver } from "@langchain/langgraph";
import { tool } from "@langchain/core/tools";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { BaseMessage, AIMessage, HumanMessage } from "@langchain/core/messages";
import { z } from "zod";
import { BedrockChat } from "@langchain/community/chat_models/bedrock";

// 1. --- GRAPH ---
const GraphState = Annotation.Root({
    input: Annotation<string>
});

const step1 = (state: typeof GraphState.State) => {
    console.log("---Step 1---");
    return state;
}

const step2 = (state: typeof GraphState.State) => {
    console.log("---Step 2---");
    return state;
}

const step3 = (state: typeof GraphState.State) => {
    console.log("---Step 3---");
    return state;
}


const builder = new StateGraph(GraphState)
    .addNode("step1", step1)
    .addNode("step2", step2)
    .addNode("step3", step3)
    .addEdge(START, "step1")
    .addEdge("step1", "step2")
    .addEdge("step2", "step3")
    .addEdge("step3", END);


// Set up memory
const graphStateMemory = new MemorySaver()

const graph = builder.compile({
    checkpointer: graphStateMemory,
    interruptBefore: ["step2"]
});


// // Input
// const initialInput = { input: "hello world" };

// // Thread
// const graphStateConfig = { configurable: { thread_id: "1" }, streamMode: "values" as const };

// // Run the graph until the first interruption
// for await (const event of await graph.stream(initialInput, graphStateConfig)) {
//     console.log(`--- ${event.input} ---`);
// }

// // Will log when the graph is interrupted, after step 2.
// console.log("---GRAPH INTERRUPTED---");

// // If approved, continue the graph execution. We must pass `null` as
// // the input here, or the graph will
// for await (const event of await graph.stream(null, graphStateConfig)) {
//     console.log(`--- ${event.input} ---`);
// }


// 2. --- AGENT ---
const AgentState = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
        reducer: (x, y) => x.concat(y),
    }),
});

const search = tool((_) => {
    return "It's sunny in San Francisco, but you better look out if you're a Gemini 😈.";
}, {
    name: "search",
    description: "Call to surf the web.",
    schema: z.string(),
})

const tools = [search]
const toolNode = new ToolNode<typeof AgentState.State>(tools)

// Set up the model
const modelWithTools = new BedrockChat({
    model: process.env.LLM_MODEL_ID,
    region: process.env.AWS_REGION,
    temperature: 0
}).bindTools(tools);


// Define nodes and conditional edges

// Define the function that determines whether to continue or not
function shouldContinue(state: typeof AgentState.State): "action" | typeof END {
    const lastMessage = state.messages[state.messages.length - 1];
    // If there is no function call, then we finish
    if (lastMessage && !(lastMessage as AIMessage).tool_calls?.length) {
        return END;
    }
    // Otherwise if there is, we continue
    return "action";
}

// Define the function that calls the model
async function callModel(state: typeof AgentState.State): Promise<Partial<typeof AgentState.State>> {
    const messages = state.messages;
    const response = await modelWithTools.invoke(messages);
    // We return an object with a messages property, because this will get added to the existing list
    return { messages: [response] };
}

// Define a new graph
const workflow = new StateGraph(AgentState)
    // Define the two nodes we will cycle between
    .addNode("agent", callModel)
    .addNode("action", toolNode)
    // We now add a conditional edge
    .addConditionalEdges(
        // First, we define the start node. We use `agent`.
        // This means these are the edges taken after the `agent` node is called.
        "agent",
        // Next, we pass in the function that will determine which node is called next.
        shouldContinue
    )
    // We now add a normal edge from `action` to `agent`.
    // This means that after `action` is called, `agent` node is called next.
    .addEdge("action", "agent")
    // Set the entrypoint as `agent`
    // This means that this node is the first one called
    .addEdge(START, "agent");


// Setup memory
const memory = new MemorySaver();

// Finally, we compile it!
// This compiles it into a LangChain Runnable,
// meaning you can use it as you would any other runnable
const app = workflow.compile({
    checkpointer: memory,
    interruptBefore: ["action"]
});


// 3. --- INTERACTING WITH AGENT ---
// Input
const inputs = new HumanMessage("search for the weather in sf now");

// Thread
const config = { configurable: { thread_id: "3" }, streamMode: "values" as const };

for await (const event of await app.stream({
    messages: [inputs]
}, config)) {
    const recentMsg = event.messages[event.messages.length - 1];
    console.log(`================================ ${recentMsg._getType()} Message (1) =================================`)
    console.log(recentMsg.content);
}

// Resume
for await (const event of await app.stream(null, config)) {
    const recentMsg = event.messages[event.messages.length - 1];
    console.log(`================================ ${recentMsg._getType()} Message (1) =================================`)
    console.log(recentMsg.content);
}

