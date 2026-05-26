# Section 1 Formatting and Generation

Obtaining an unstructured text from a large language model (LLM) often does not meet practical needs in applications. In order to implement more complex logic, interact with external tools, or present data in a user-friendly way, the model needs to be able to output data with a specific structure, such as JSON or XML.

This section will discuss several mainstream methods for formatting generation, including solutions built into frameworks such as LangChain and LlamaIndex, implementation ideas that do not rely on frameworks, and a more powerful technology - Function Calling.

> In the generation phase, prompt word engineering is also an important part. However, since there has been a lot of introduction in the previous chapters, I will not go into details in this chapter.

## 1. Why is formatting required?

Let’s first look at some specific application scenarios:

- **RAG-driven e-commerce customer service**: When a user asks "Recommend several keyboards suitable for programmers", we hope that LLM will return a JSON list containing product names, prices, features and purchase links, instead of a piece of descriptive text, so that the front end can directly render it into a product card.
- **Natural language to API call**: The user says "Help me check the flights from Shanghai to Beijing tomorrow." The system needs to parse this sentence into a structured API request, such as`{"departure": "上海", "destination": "北京", "date": "2025-07-18"}`.
- **Automatic data extraction**: Automatically extract key information such as events, time, location, and involved people from a news article, and store it in a structured form in the database.

In these scenarios, formatting generation is the key link between the LLM's natural language understanding capabilities and the programmatic logic of the downstream application.

## 2. Implementation method of format generation

> [Full Code](https://github.com/datawhalechina/all-in-rag/blob/main/code/C5/01_pydantic.py)

### 2.1 Output Parsers

LangChain provides a powerful component -`OutputParsers`(output parser), which is specially used to process the output of LLM. The main idea is to automatically inject an instruction on how to format the output in the prompt (Prompt) sent to LLM, and after getting the result, parse the plain text string returned by LLM into expected structured data (such as Python objects).

LangChain provides a variety of parsers out of the box, such as:

- **StrOutputParser**: The most basic output parser, which simply returns the output of LLM as a string.
- **JsonOutputParser**: Can parse complex JSON strings containing nested structures and lists.
- **PydanticOutputParser**: Combined with Pydantic models, the most stringent definition and verification of output formats can be achieved.

Next, we will focus on analyzing the working principle of`PydanticOutputParser`through a specific code example. It guides LLM to produce JSON output that strictly conforms to that data structure by converting a user-defined Pydantic data model into detailed formatting instructions and injecting them into prompt words. Finally, the JSON string returned by the model is safely parsed into a Pydantic object instance.

```python
# (此处省略了导入和 LLM 初始化代码)

# 1. 定义期望的数据结构
class PersonInfo(BaseModel):
    """用于存储个人信息的数据结构。"""
    name: str = Field(description="人物姓名")
    age: int = Field(description="人物年龄")
    skills: List[str] = Field(description="技能列表")

# 2. 基于 Pydantic 模型，创建解析器
parser = PydanticOutputParser(pydantic_object=PersonInfo)

# 3. 创建提示模板，注入格式指令
prompt = PromptTemplate(
    template="请根据以下文本提取信息。\n{format_instructions}\n{text}\n",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 4. 创建处理链 (假定 llm 已被初始化)
chain = prompt | llm | parser

# 5. 执行调用
text = "张三今年30岁，他擅长Python和Go语言。"
result = chain.invoke({"text": text})

# 6. 打印结果
print(result)
# name='张三' age=30 skills=['Python', 'Go语言']
```

(1) **Define data model**: Use Pydantic's`BaseModel`to define the`PersonInfo`class. This is not only a Python object, but also a clear data structure specification (Schema). The`description`description text in`Field`will be provided directly as instructions to the large model, so its expression needs to be clear and precise.

(2) **Generate format instructions**: When`PydanticOutputParser`is instantiated, its`get_format_instructions()`method will perform the following operations:
- Call the`.model_json_schema()`method of the Pydantic model to extract the JSON Schema definition of the data structure.
- Simplify the Schema and embed it into a preset, guided prompt template. This template explicitly requires LLM to output a JSON object that conforms to this Schema.

(3) **Build and execute the call chain**: Link`prompt`,`llm`and`parser`through LangChain Expression Language (LCEL). When the call chain is triggered:
-`prompt`will combine the user input (`text`) and the format command (`format_instructions`) generated in the previous step into the final prompt and send it to`llm`.
-`llm`Generates a JSON-formatted string based on this prompt, which contains strict formatting requirements.

(4) **Parsing and Verification**: After`PydanticOutputParser`receives the string returned by LLM, it will perform a two-step parsing process:
- First, it inherits from`JsonOutputParser`and will parse the text string output by LLM into a Python dictionary.
- Then, the most critical step, it will use the`PersonInfo.model_validate()`method to verify the dictionary with the defined data model. If the dictionary's key and value types conform to the definition of`PersonInfo`, the parser will return an instance object of`PersonInfo`; if validation fails, a`OutputParserException`exception will be thrown.

### 2.2 Output analysis of LlamaIndex

LlamaIndex's output parsing and generation processes are closely integrated, mainly reflected in its two core components, namely Response Synthesis and Structured Output.

In the RAG process, after the retriever recalls a series of related text blocks (Nodes), it does not simply splice them together. The Response Synthesizer is responsible for taking these chunks of text and the original query and presenting them to the LLM in a smarter way to generate the final answer. For example, it can process information chunk by chunk and iteratively refine the answer (`refine`mode), or compress as many text chunks as possible into a single LLM call (`compact`mode). The default goal of this stage is to generate a high-quality **text** answer.

LlamaIndex mainly uses **Pydantic Programs** when LLM is required to return structured data (such as JSON) instead of plain text. This is consistent with LangChain’s`PydanticOutputParser`idea:

- **Define Schema**: Developers first define a Pydantic model and clarify the data structure, fields and types required for output.
- **Guide generation**: LlamaIndex will convert this Pydantic model into instructions in a format that LLM can understand. If the underlying LLM supports Function Calling, LlamaIndex will preferentially use this feature to obtain more reliable structured output. If not, it falls back to injecting the JSON Schema into the prompt.
- **Parse Verification**: Finally, the output returned by LLM is automatically parsed and verified with the Pydantic model to ensure that its type and structure are completely correct, and finally a Pydantic object instance is returned.

### 2.3 Simple implementation ideas that do not rely on frameworks

If you don't want to rely on a specific framework, you can also implement formatted generation through the tips of the project.

The main idea is to give clear, unambiguous instructions and examples in the prompts. Here are some practical tips:

- **Explicitly require JSON format**: Directly and forcefully require the model "must return a JSON object" and "do not include any explanatory text, only return JSON" in the prompt.
- **Provide JSON Schema**: Provide the schema (Schema) of the JSON object you want in the prompt, describing the meaning and data type of each key.
- **Provide few-shot examples**: Give 1-2 complete examples of "user input -> expected JSON output" to let the model learn the format and style of the output.
- **Use syntax constraints**: For some locally deployed open source models (such as models run through`llama.cpp`), you can use syntax files such as GBNF (GGML BNF) to force the output of the model to ensure that each token it generates strictly conforms to the predefined JSON syntax. This is the strictest and most reliable non-Function Calling method.

## 3. Function Calling

Function Calling (or Tool Calling) is an important development in the field of LLM in recent years, improving the model's ability to interact with the external world and generate structured data.

> [Complete code of this section](https://github.com/datawhalechina/all-in-rag/blob/main/code/C5/02_function_calling_example.py)

### 3.1 Concept and Workflow

The essence of Function Calling is a multi-turn conversational process that allows models, code, and external tools (such as APIs) to work together. Its core workflow is as follows:

(1) **Define tools**: First, define the available tools in a specific format (usually JSON Schema) in the code, including the name of the tool, function description, and required parameters.

(2) **User question**: The user initiates a request that requires calling a tool to answer.

(3) **Model Decision**: After receiving the request, the model analyzes the user's intention and matches the most appropriate tool. It does not answer directly, but returns a special response containing`tool_calls`. This response is equivalent to an instruction: "Please call such-and-such tool and use these parameters."

(4) **Code execution**: The application receives this instruction, parses out the tool name and parameters, and then **actually executes** the tool at the code level (for example, calling a real weather API).

(5) **Result feedback**: Pack the execution result of the tool (for example, the real weather data obtained from the API) into a message with`role`as`tool`, and send it to the model again.

(6) **Final generation**: After receiving the execution results of the tool, the model combines the original question and the information returned by the tool to generate a final, natural language answer.

### 3.2 Function Calling Practice

Next, use the example of`openai`directly to demonstrate the above process.


```python
# 1. 定义工具
tools = [...] 

# 2. 用户提问
messages = [{"role": "user", "content": "杭州今天天气怎么样？"}]
message = send_messages(messages, tools=tools)

# 3. 代码执行：模拟调用天气API，并将结果添加到消息历史
if message.tool_calls:
    tool_call = message.tool_calls[0]
    messages.append(message) # 添加模型的回复
    tool_output = "24℃，晴朗" # 模拟API结果
    messages.append({
        "role": "tool", 
        "tool_call_id": tool_call.id, 
        "content": tool_output
    }) # 添加工具执行结果

    # 4. 第二次调用 (`Tool -> Model`)：将工具结果返回给模型，获取最终回答
    final_message = send_messages(messages, tools=tools)
    print(final_message.content)
```

Key steps:

(1) **Definition`tools`**: Contains all available function definitions in a list. Each definition is a JSON object that strictly describes the function's name (`name`), functionality (`description`), and parameters (`parameters`). The quality of this description directly determines whether the model can correctly select and use tools.

(2) **First call (`User -> Model`)**: Send the user’s original question (`"role": "user"`) and the`tools`list to the model.

(3) **Processing`tool_calls`**: Check whether the model's response contains`tool_calls`. If included, it indicates that the model decided to use the tool. Parse out the function name and parameters, and simulate execution (in a real scenario, this will be a real API call).

(4) **Second call (`Tool -> Model`)**: Pack the original user question, the model's tool call response, and the tool result (`"role": "tool"`) obtained after the simulation execution into a new conversation history, and send it to the model again.

(5) **Get the final answer**: After seeing the execution results of the tool, the model can answer the user's initial question in natural language.

### 3.3 Advantages of Function Calling

Compared with simply outputting JSON by prompting the project to "request" the model, the advantages of Function Calling are:

- **Higher reliability**: This is a capability natively supported by the model. Compared with parsing plain text output that may be in unstable format, the structured data obtained in this way is more stable and accurate.
- **Intent recognition**: It is not only formatted output, but also includes "mapping of intentions to functions". The model proactively selects the most appropriate tool based on the user's problem.
- **Interaction with the external world**: It is the core foundation for building AI agents that can perform actual tasks, allowing LLM to query databases, call APIs, control smart homes, etc.