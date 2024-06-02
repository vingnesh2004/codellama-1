import os 
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.llms import Ollama
import streamlit as st



# Pull the model if needed (assuming it's available in Langchain)
# Ollama.pull("Code Llama")  # Uncomment this line if you need to pull the model

# Define a custom output parser


class CustomOutputParser:
    def __call__(self, text: str):
        return text

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's queries."),
        ("user", "Question: {question}")
    ]
)

# Streamlit framework
st.title('A.L.B.U.S With Code Llama API')
input_text = st.text_input("Search the topic you want")

# Define the LLM instance
llm = Ollama(model="codellama")
output_parser = CustomOutputParser()  # Use the custom output parser

# Compose the chain
def run_chain(question):
    prompt_text = prompt.format(question=question)
    response = llm(prompt_text)
    return output_parser(response)

# Handle user input
if input_text:
    response = run_chain(input_text)
    st.write(response)
