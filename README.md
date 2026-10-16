# ClickCar Chat Agents 🚗💬

A Streamlit prototype for helping an auto-parts salesperson turn a chat request into inventory suggestions. The current entry is [streamlit_app.py](streamlit_app.py); `streamlit_app_legacy.py` is a separate, unreviewed alternate.

This handoff describes static source at `9acc42877967541d8d3613f7752037e0b7d25b30`. It does not certify a working install, inventory compatibility, provider access, or current model availability.

## Setup route and limits

Use the existing [requirements.txt](requirements.txt) rather than the original four-package example:

~~~bash
pip install -r requirements.txt
~~~

The manifest is unpinned and declares no Python version. The entry imports `numpy` and `requests` without explicit entries for them in the manifest; they may arrive transitively. Dependency resolution, LlamaIndex/OpenAI API compatibility, and installation have not been verified.

Before starting, distinguish these configuration paths. No secret values belong in this README.

| Configuration | What the current source does |
| --- | --- |
| OpenAI SDK default credentials | Constructs `OpenAI()` without an explicit key before sidebar interaction. Its default credentials, normally the `OPENAI_API_KEY` environment variable, must be available before that constructor. This setup has not been exercised. |
| Streamlit secrets `url1` and `url3` | Reads both at module load. Exporting an environment variable named `url1` does not describe these `st.secrets` reads. The visible chat handler uses the local workbook route below. |
| Streamlit secret `OPENAI_API_KEY` | Reads it later inside retrieval and assigns `OpenAI.api_key`; this is separate from the earlier client construction. |
| Streamlit secret `COHERE_API_KEY` | Supplies the Cohere reranking client during retrieval. |

The sidebar key input exists, but only stores `session_state["openai_api_key"]`. That value is not passed to the precreated client, and the sidebar does not change the model assignment. Do not treat sidebar entry as sufficient credential setup.

The declared entry command is:

~~~bash
streamlit run streamlit_app.py
~~~

This is the corrected launcher, not a recorded successful run. Configuration, the inventory workbook, embedding assets, and dependency compatibility still need separate validation.

## Current source path

The visible chat handler calls OpenAI completions, reads the local Excel inventory workbook, prepares embedding assets if its local directory is absent, performs LlamaIndex/BM25 retrieval and Cohere reranking, then uses retrieved content in further OpenAI calls and summaries. The workbook is tracked, but its contents/schema were not opened in this review. The embedding directory is not tracked; its preparation/save path has not been run.

The source unconditionally assigns `gpt-3.5-turbo-1106` before chat, overriding an earlier conditional GPT-4 default. This is a historical identifier in the code, not a claim that a provider currently serves it. HTTP inventory helpers remain defined, but the visible chat handler calls the local retrieval function instead.

Initial UI rendering also calls a translation helper. Chat/provider calls can send user requests and inventory-derived text externally; embedding preparation can create local assets. These are application actions, not offline documentation checks. None were exercised for this correction.

## Original product intent

The original description and feature list below record product intent. They do not override the current-source limits above:

> ClickCar Chat Agents is a chatbot application that uses GPT-4 to assist users in finding auto parts by summarizing requests, translating queries, and fetching detailed information from a database.

### Intended features ✨
1. Translates queries to Spanish 🇪🇸 to cater to a wider audience.
2. Summarizes user needs to manage the auto part shopping experience efficiently 📋.
3. Specializes in picking auto parts based on the user's summarized memory 💾.
4. Retrieves relevant auto part details from a query and an inventory database 🗂️.
5. Provides an interactive chat interface for users to express their auto part needs 💬.
6. Employs a tailored system prompt to guide the chat agent in assisting the users 📘.
7. Offers a user-friendly Streamlit interface to display auto parts inventory and chat interactions 🖥️.

The superseded setup and full original wording remain available in the [immutable original README](https://github.com/HomenShum/clickcar_salesagent_with_custom_autogen/blob/9acc42877967541d8d3613f7752037e0b7d25b30/README.md). This correction retains the product intent without presenting that older setup as current runnable instructions.

## Contributing 🤝
Contributions to improve ClickCar Chat Agents are welcome. Please ensure to update tests as appropriate.

## License 📜
Restricted to usage by whoever approved by Homen Shum via official contract 🚫📄.

## Documentation history

See the [README history](CHANGELOG/docs/readme.md) for this source-only correction and its verification limits.
