# lammy-v1

Lammy is an LLM-based app that helps students learn more from examiner feedback. Play with it here: https://lammy-v1.streamlit.app/

I named it after my brother, Lammy, and built it for him. We are first-generation university students and for many other reasons he's had a tough time getting to where he is. AI technology has the potential to help establish more equal access to education for students all over the world. This app is one small step in testing out how to make that happen. The idea is that students who may be nervous to ask for more clarity on feedback or who want more actionable advice can get what they need to make progress on this app. It is not intended to replace any part of the established education system, just to supplement it for those who need it. 

The app is currently in POC form and is built on Streamlit for ease of testing these ideas. In future iterations I plan to build a custom UI. The backend currently uses a GPT language model and retreiveal augmented generation (RAG) with LangChain libaries to ingest three user input (essay, mark, examiners feedback) and produce output. Again in future iterations I plan to create more structured output and get the RAG working better for a more personalised user experience. 

Thanks for checking it out! 
