## clickcar_salesagent_with_custom_autogen 🚗💬
ClickCar Chat Agents is a chatbot application that uses GPT-4 to assist users in finding auto parts by summarizing requests, translating queries, and fetching detailed information from a database.

## ClickCar Chat Agents 🤖
ClickCar Chat Agents is a Streamlit app designed to enhance the auto parts shopping experience. By leveraging the GPT-4 model, it assists users in finding the best auto parts based on their needs and preferences.

## Features ✨
1. Translates queries to Spanish 🇪🇸 to cater to a wider audience.
2. Summarizes user needs to manage the auto part shopping experience efficiently 📋.
3. Specializes in picking auto parts based on the user's summarized memory 💾.
4. Retrieves relevant auto part details from a query and an inventory database 🗂️.
5. Provides an interactive chat interface for users to express their auto part needs 💬.
6. Employs a tailored system prompt to guide the chat agent in assisting the users 📘.
7. Offers a user-friendly Streamlit interface to display auto parts inventory and chat interactions 🖥️.

## Requirements 🛠️
1. OpenAI API Key 🔑
2. Streamlit 🌟
3. Pandas 🐼
4. Requests 📤
5. Deep Translator 🌐
6. Environment Variables 🌍
7. url1: Endpoint URL for the auto parts inventory database, to be set in your environment.

## Installation 💻
Before running the app, ensure you have Python installed and then install the required packages:

~~~bash
pip install streamlit pandas requests deep_translator
~~~
Set the url1 environment variable:
~~~bash
export url1='your_inventory_database_endpoint'
~~~
Usage 📝
Run the app using Streamlit:

~~~bash
streamlit st_streaming_query_display.py
~~~
Input your OpenAI API Key in the sidebar when prompted. The app will then be ready to assist with auto parts selection.

## How It Works 🧩
The application follows these steps:

1. Translates the user query to Spanish for broader accessibility 🌐.
2. Summarizes the user's memory for a streamlined shopping experience 🧠.
3. Uses a system prompt to guide the chat assistant's interactions 💁‍♂️.
4. Retrieves auto parts details from the database based on user queries 🔍.
5. Engages with the user in a conversational manner, including playful chit-chat mimicking an Italian gangster movie character to spice up the mood 🎥🍝.

## Streamlit UI 🎛️
1. Chat interface for user queries and assistant responses 💻.
2. Sidebar for API key input and model details 🔑.
3. Inventory database preview in the sidebar 📊.
4. Button to clear messages and reset the conversation 🔄.

## Contributing 🤝
Contributions to improve ClickCar Chat Agents are welcome. Please ensure to update tests as appropriate.

## License 📜
Restricted to usage by whoever approved by Homen Shum via official contract 🚫📄.

This README provides a comprehensive guide to your Streamlit app, including features, installation steps, usage instructions, and contribution guidelines. Adjust the file paths and commands according to your repository's structure and personal setup.
