import streamlit as st
from utils import retrieve_doc
from utils import feedback_doc
from annotated_text import annotated_text
import st_state_patch

def annotate_answer(answer,context):
    start_idx = context.find(answer)
    end_idx = start_idx+len(answer)
    annotated_text(context[:start_idx],(answer,"ANSWER","#8ef"),context[end_idx:])
      
st.write("# Haystack Demo")
st.sidebar.header("Options")
top_k_reader = st.sidebar.slider("Max. number of answers",min_value=1,max_value=10,value=3,step=1)
top_k_retriever = st.sidebar.slider("Max. number of documents from retriever",min_value=1,max_value=10,value=3,step=1)
question = st.text_input("Please provide your query:",value="Who is the father of Arya Starck?")
s = st.State()
if s and s.run_query:
    run_query = s.run_query
    st.button("Run")
else:
    run_query = st.button("Run")
s.run_query = run_query

debug = st.sidebar.checkbox("Show debug info")
raw_json_feedback = ""
if run_query:
    with st.spinner("Performing neural search on documents... 🧠 \n "
                    "Do you want to optimize speed or accuracy? \n"
                    "Check out the docs: https://haystack.deepset.ai/docs/latest/optimizationmd "):
        results,raw_json = retrieve_doc(question,top_k_reader=top_k_reader,top_k_retriever=top_k_retriever)
    st.write("## Retrieved answers:")
    for result in results:
        annotate_answer(result['answer'],result['context'])
        '**Relevance:** ', result['relevance'] , '**Source:** ' , result['source']
        if st.button("Correct answer", key=result['answer']):
           st.write("Correct answer")
           raw_json_feedback = feedback_doc(question,"true",result['document_id'],1,"true",result['answer'],result['offset_start_in_doc'])
        if st.button("Wrong answer and wrong passage", key=result['answer']):
           st.write("Wrong answer and wrong passage")
           raw_json_feedback = feedback_doc(question,"false",result['document_id'],1,"false",result['answer'],result['offset_start_in_doc'])
        if st.button("wrong answer, but correct passage", key=result['answer']):
           st.write("wrong answer, but correct passage")
           raw_json_feedback = feedback_doc(question,"false",result['document_id'],1,"true",result['answer'],result['offset_start_in_doc'])
    if debug:
        st.subheader('REST API JSON response')
        st.write(raw_json)
if debug:
   st.write(raw_json_feedback)
   
