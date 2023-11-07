from openai import OpenAI
import openai
import streamlit as st
import pandas as pd
import copy
import requests
from deep_translator import GoogleTranslator
import json
import re 
import os
#################### Settings ####################

client = OpenAI()

url1 = st.secrets['url1']
# url1 = os.environ['url1']

def translate_query(text):
    translated_text = GoogleTranslator(source='english', target='spanish').translate(text)
    return translated_text

@st.cache_data
def memory_summary_agent(memory):
    summarization = client.chat.completions.create(
        model = st.session_state["openai_model"],
        messages=[
            {"role": "system", "content": 'You specialize at summarizing and keeping track of the user needs to help manage auto part shopping experience'},
            {"role": "user", "content": memory},
        ],
    )
    text = summarization.choices[0].message.content.strip()
    text = re.sub("\s+", " ", text)
    return text

@st.cache_data
def auto_part_picking_agent(memory):
    """
    Function that specializes in picking auto parts based on the user's summarized memory.
    
    :param memory: A string containing the summarized memory of user's needs and preferences.
    :return: A string with the model's response, aimed at assisting with auto parts selection.
    """
    try:
        summarization = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": "system", "content": 'You are an assistant specializing in auto parts. Your task is to assist the user in finding the best auto parts based on their needs, preferences, and past interactions.'},
                {"role": "user", "content": memory},
            ],
        )
        text = summarization.choices[0].message.content.strip()
        text = re.sub("\s+", " ", text)
        return text
    except Exception as e:
        # You may want to handle exceptions in a way that is appropriate for your application.
        st.error("An error occurred while trying to pick auto parts: {}".format(e))
        return None

@st.cache_data
def summarize_all_messages(message):
    return memory_summary_agent(". ".join(st.session_state.memory) + ". " + message)

@st.cache_data
def retrieve_auto_parts_details(query):
    """
    Retrieve the most relevant auto part details based on a given query.

    :param url: The endpoint URL to make the request to.
    :param query: The search query to find auto parts.
    :return: A dictionary containing a list of auto parts with their details.
    """
    url = st.secrets['url1']
    # url = os.environ['url1']
    try:
        response = requests.get(url, params={'q': query})
        response.raise_for_status()
        # We expect the response to be a JSON with a 'results' field containing the parts details
        part_details = response.json().get('results', [])
        st.session_state.df = pd.DataFrame(part_details)
        with st.sidebar:
            st.dataframe(st.session_state.df)
        st.session_state.memory.append(memory_summary_agent(". New data pulled from database that may or may not match user needs: " + str(part_details) + ". Previous information regarding user needs: " + st.session_state.memory[-1]))
        return {'part_details': part_details}
    except requests.exceptions.RequestException as e:
        # This will capture any errors related to the request
        return {'error': str(e)}

# Set the system and user prompts

salesperson_system_prompt = (
    "Concise Response. Customer Orientation: Prioritize the customer's needs and recommend products that align with these needs.\n"
    "Politeness and Education: Always be polite, patient, and courteous to create a welcoming atmosphere.\n"
    "Good Listening: Pay attention to the customer's needs and preferences for personalized recommendations.\n"
    "Creativity and Adaptability: Be creative and adaptable in offering innovative solutions to various situations.\n"
    "Product Knowledge: Have an in-depth understanding of the products being sold for informed recommendations.\n"
    "Positive Attitude: Maintain a positive and smiling demeanor to enhance customer experience.\n"
    "Problem-Solving Skills: Address problems calmly and find appropriate solutions rather than being rude.\n"
    "Enjoyment of Helping: Take pleasure in assisting customers and show patience, especially during lengthy decision-making processes.\n"
    "Proactivity: Taking the initiative to assist customers is beneficial, although not essential.\n"
    "Enjoyment of People Contact: Enjoy direct interaction with people and be willing to build relationships with customers.\n\n"
    "In essence, a successful salesperson or conversational assistant like \"MAN-Ai\" should aim to provide a positive, personalized experience by understanding customer needs and offering friendly, effective solutions.\n\n"
    "FOLLOW THESE STEPS:\n"
    "1. Ask questions to assist users in finding the right auto parts.\n"
    "2. Extract relevant auto part details from their queries, feed into inventory database query to return inventory extracted details.\n"
    "3. Quote back to user's message, recommend auto parts based on the user's messages, extracted details and available inventory.\n"
    "4. Occasionally ask user to see if they found the item that they needed. \n"
    "5. Occasionally review the new information from inventory database to see if it matches user need given the previous information and user prompt.\n"
    "6. Chit Chat with the user like an italian gangster from the movies to spice up the mood."
)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4-1106-preview"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = []

if "auto_part_criteria" not in st.session_state:
    st.session_state.auto_part_criteria = []

if "auto_part_details" not in st.session_state:
    st.session_state.auto_part_details = []
    
st.session_state.memory.append("")

st.session_state["openai_model"] = "gpt-3.5-turbo-1106"
st.session_state["openai_model_context_window"] = "16K Tokens"
st.session_state["openai_model_training_data"] = "Up to Sep 2021"

#################### Main ####################
### OpenAI API Key Management ###
selected_key = None
with st.sidebar:
    st.header("Input your OpenAI API Key")
    selected_key = st.text_input("API Key", type="password")
    with st.sidebar:
        st.subheader("Using OpenAI Model:")
        st.write(st.session_state["openai_model"])

        st.subheader("Context Window: ")
        st.write(st.session_state["openai_model_context_window"])

        st.subheader("Training Data: ")
        st.write(st.session_state["openai_model_training_data"])


if not re.match(r"sk-\S+", selected_key):
    with st.sidebar:
        st.warning("Use your own OpenAI API key for full GPT-4-Turbo Experiece. The context window and training data are improved to 128K Tokens and Up to Apr 2023.")
    # openai.api_key = os.environ['OPENAI_API_KEY']
    openai.api_key = st.secrets['OPENAI_API_KEY']
else:
    st.session_state["openai_api_key"] = selected_key
    openai.api_key = st.session_state["openai_api_key"]
    st.session_state["openai_model"] = "gpt-4-turbo-1106"
    st.session_state["openai_model_context_window"] = "128K Tokens"
    st.session_state["openai_model_training_data"] = "Up to Apr 2023"
### END OpenAI API Key Management END ###

### Streamlit UI ###
st.title("ClickCar Chat Agents")
st.subheader("What auto parts are you looking for from ClickCar store?\n")
st.subheader(translate_query("What auto parts are you looking for from ClickCar store?"))

###
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
###

with st.sidebar:
    st.header("Preview inventory database")

with st.container():
    if prompt := st.chat_input("Any auto part that you are looking for in ClickCar?"):
        st.session_state.memory.append(memory_summary_agent(". New information from User about user needs: " + prompt + ". Previous information: " + st.session_state.memory[-1]))
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            full_spanish_response = ""
            for response in client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": "system", "content": salesperson_system_prompt},
                    {"role": "user", "content": ". Previous information: "+ st.session_state.memory[-1]},
                    {"role": "user", "content": ". New user prompt: "+ prompt},
                    {"role": "user", "content": ". New information from inventory database: "+ str(st.session_state.auto_part_details)},
                ],
                stream=True,
            ):
                full_response += str(response.choices[0].delta)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response + "/// SPANISH /// " + str(translate_query(full_response)))
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.memory.append(memory_summary_agent(". New information from Assistant about user needs: " + full_response + ". Previous information: " + st.session_state.memory[-1]))
        total_memory = st.session_state.memory[-1]
        auto_part_picking_agent_response = auto_part_picking_agent(total_memory)
        st.session_state.auto_part_criteria.append(auto_part_picking_agent_response)
        retrieve_auto_parts_details_response = retrieve_auto_parts_details(auto_part_picking_agent_response)
        st.session_state.auto_part_details.append(retrieve_auto_parts_details_response)

if st.button("Clear Messages"):
    st.session_state.messages = []
    st.session_state.memory = []
    st.rerun()
