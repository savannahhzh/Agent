import "dotenv/config";
import { tool } from "@langchain/core/tools";
import { HumanMessage, AIMessage, SystemMessage, ToolMessage } from "@langchain/core/messages";
import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";

const systemPrompt = `You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location.`;

const getWeatherForLocation = tool(
  (input) => `It's always sunny in ${input.city}!`,
  {
    name: "get_weather_for_location",
    description: "Get the weather for a given city",
    schema: z.object({
      city: z.string().describe("The city to get weather for"),
    }),
  }
);

const getUserLocation = tool(
  (_, config) => {
    const userId = (config?.configurable as { user_id?: string })?.user_id;
    return userId === "1" ? "Florida" : "SF";
  },
  {
    name: "get_user_location",
    description: "Retrieve the current user's location based on their user ID",
    schema: z.object({}),
  }
);

const tools = [getWeatherForLocation, getUserLocation];
const toolMap = new Map(tools.map((t) => [t.name, t]));

const model = new ChatOpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: { baseURL: "https://api.deepseek.com/v1" },
  modelName: "deepseek-chat",
}).bindTools(tools);

async function runAgent(userMessage: string, userId = "1") {
  const messages: (HumanMessage | AIMessage | SystemMessage | ToolMessage)[] = [
    new SystemMessage(systemPrompt),
    new HumanMessage(userMessage),
  ];

  while (true) {
    const response = await model.invoke(messages);
    messages.push(response as AIMessage);

    if (!response.tool_calls || response.tool_calls.length === 0) {
      return response.content;
    }

    for (const toolCall of response.tool_calls) {
      const toolFn = toolMap.get(toolCall.name);
      if (!toolFn) throw new Error(`Unknown tool: ${toolCall.name}`);

      const result = await toolFn.invoke(toolCall.args, {
        configurable: { user_id: userId },
      });

      messages.push(new ToolMessage({ tool_call_id: toolCall.id!, content: String(result) }));
    }
  }
}

const answer = await runAgent("What's the weather where I am?", "1");
console.log("\n🤖 AI 响应：", answer);
