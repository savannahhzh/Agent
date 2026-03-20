import "dotenv/config"; // 加载配置文件 .env
import { ChatOpenAI } from "@langchain/openai";
import { AIMessage, ToolMessage, HumanMessage} from "langchain";

const aiMessage = new AIMessage({
  content: [],
  tool_calls: [{
    name: "get_weather",
    args: { location: "San Francisco" },
    id: "call_123"
  }]
});

const toolMessage = new ToolMessage({
  content: "Sunny, 72°F",
  tool_call_id: "call_123"
});

const messages = [
  new HumanMessage("What's the weather in San Francisco?"),
  aiMessage,  // Model's tool call
  toolMessage,  // Tool execution result
];


const model = new ChatOpenAI({ 
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: { baseURL: "https://api.deepseek.com/v1" },
  modelName: "deepseek-chat",
})

const response = await model.invoke(messages);  // Model processes the result
console.log(response); 
// AIMessage {
//   "id": "4ad732c7-1673-46ec-91ff-c191e7f2fc44",
//   "content": "It's currently **sunny and 72°F** in San Francisco.",
//   "additional_kwargs": {}, // 原始厂商数据
//   "response_metadata": { // Token 消耗、停止原因
//     "tokenUsage": {
//       "promptTokens": 81,
//       "completionTokens": 16,
//       "totalTokens": 97
//     },
//     "finish_reason": "stop",
//     "model_provider": "openai",
//     "model_name": "deepseek-chat",
//     "usage": {
//       "prompt_tokens": 81,
//       "completion_tokens": 16,
//       "total_tokens": 97,
//       "prompt_tokens_details": {
//         "cached_tokens": 0
//       },
//       "prompt_cache_hit_tokens": 0,
//       "prompt_cache_miss_tokens": 81
//     },
//     "system_fingerprint": "fp_eaab8d114b_prod0820_fp8_kvcache_new_kvcache"
//   },
//   "tool_calls": [],  如果 AI 想调工具，这里会有数组
//   "invalid_tool_calls": [],
//   "usage_metadata": {  计费信息
//     "output_tokens": 16,
//     "input_tokens": 81,
//     "total_tokens": 97,
//     "input_token_details": {
//       "cache_read": 0
//     },
//     "output_token_details": {}
//   }
// }