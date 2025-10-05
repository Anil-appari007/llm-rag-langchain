# from langchain_openai import OpenAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

model = init_chat_model("gpt-4o-mini", model_provider="openai")
# print(model.invoke("what is the time in India?").content)


messages = [
    SystemMessage(content="Translate the following from English into French"),
    HumanMessage(content="hi!"),
]
# print(model.invoke(messages).content)


messages = [
    SystemMessage(content="Translate the following from English into Russian"),
    HumanMessage(content="Good Morning"),
]

# for token in model.stream(messages):
#     print(token.content, end="|")


system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})

prompt
print(prompt.to_messages())
print(model.invoke(prompt).content)