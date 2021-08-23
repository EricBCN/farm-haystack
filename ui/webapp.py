import os
import sys

import pandas as pd
import streamlit as st
from annotated_text import annotated_text

# streamlit does not support any states out of the box. On every button click, streamlit reload the whole page
# and every value gets lost. To keep track of our feedback state we use the official streamlit gist mentioned
# here https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
import SessionState
from utils import feedback_doc
from utils import retrieve_doc
from utils import upload_doc

# Adjust to a question that you would like users to see in the search bar when they load the UI:
DEFAULT_QUESTION_AT_STARTUP = "Who is the father of Arya Stark?"


def annotate_answer(answer, context):
    """ If we are using an extractive QA pipeline, we'll get answers
    from the API that we highlight in the given context"""
    start_idx = context.find(answer)
    end_idx = start_idx + len(answer)
    # calculate dynamic height depending on context length
    height = int(len(context) * 0.50) + 5
    annotated_text(context[:start_idx], (answer, "ANSWER", "#8ef"), context[end_idx:], height=height)

def show_plain_documents(text):
    """ If we are using a plain document search pipeline, i.e. only retriever, we'll get plain documents
    from the API that we just show without any highlighting"""
    st.markdown(text)


def random_questions(df):
    """
    Helper to get one random question + gold random_answer from the user's CSV 'eval_labels_example'.
    This can then be shown in the UI when the evaluation mode is selected. Users can easily give feedback on the
    model's results and "enrich" the eval dataset with more acceptable labels
    """
    random_row = df.sample(1)
    random_question = random_row["Question Text"].values[0]
    random_answer = random_row["Answer"].values[0]
    return random_question, random_answer


# Define state
state_question = SessionState.get(
    random_question=DEFAULT_QUESTION_AT_STARTUP, random_answer="", next_question="false", run_query="false",
    data_frame=pd.DataFrame({'Question': [""],'Answer1': [""],'Answer2': [""],'Answer3': [""],'Answer4': [""],}),
    question_indices=0
)

# Initialize variables
eval_mode = False
random_question = DEFAULT_QUESTION_AT_STARTUP
eval_labels = os.getenv("EVAL_FILE", "eval_labels_example.csv")

# UI search bar and sidebar
st.write("# Haystack Demo")
st.sidebar.header("Options")
top_k_reader = st.sidebar.slider("Max. number of answers", min_value=1, max_value=10, value=3, step=1)
top_k_retriever = st.sidebar.slider(
    "Max. number of documents from retriever", min_value=1, max_value=10, value=3, step=1
)
eval_set_mode = st.sidebar.checkbox("Evaluation set creation")  
eval_mode = st.sidebar.checkbox("Evaluation mode")
debug = st.sidebar.checkbox("Show debug info")

st.sidebar.write("## File Upload:")
data_files = st.sidebar.file_uploader("", type=["pdf", "txt", "docx"], accept_multiple_files=True)
for data_file in data_files:
    # Upload file
    if data_file:
        raw_json = upload_doc(data_file)
        st.sidebar.write(raw_json)
        if debug:
            st.subheader("REST API JSON response")
            st.sidebar.write(raw_json)

# load csv into pandas dataframe
if eval_mode:
    try:
        df = pd.read_csv(eval_labels, sep=";")
    except Exception:
        sys.exit("The eval file was not found. Please check the README for more information.")
    if (
        state_question
        and hasattr(state_question, "next_question")
        and hasattr(state_question, "random_question")
        and state_question.next_question
    ):
        random_question = state_question.random_question
        random_answer = state_question.random_answer
    else:
        random_question, random_answer = random_questions(df)
        state_question.random_question = random_question
        state_question.random_answer = random_answer

# Get next random question from the CSV
if eval_mode:
    next_question = st.button("Load new question")
    if next_question:
        random_question, random_answer = random_questions(df)
        state_question.random_question = random_question
        state_question.random_answer = random_answer
        state_question.next_question = "true"
        state_question.run_query = "false"
    else:
        state_question.next_question = "false"

# Create table for evaluation set
if eval_set_mode:
# Intro to good questions
    st.write("How to create good questions for your evaluation set")
    good_question_expander = st.beta_expander("What is a good question?", expanded=False)
    with good_question_expander:
        st.write('- A good question is a fact-seeking question that can be answered with an entity (person, organisation, location, etc.) or explanation. A bad question is ambiguous, incomprehensible, dependent on clear false presuppositions, opinion seeking, or not clearly a request for factual information.')
        st.write('- The question should ask about information present in the text passage given. It should not be answerable only with additional knowledge or your interpretation.')
        st.write('- Do not copy paste answer text into the question. Good questions do not contain the exact same words as the answer or the context around the answer. The question should be a reformulation with synonyms and in different order as the context of the answer.')
        st.write('- Questions should be very precise natural questions you would ask when you want information from another person.')
    how_many_expander = st.beta_expander("How many questions should you ask per text passage?", expanded=False)
    with how_many_expander:
        st.write('- Maximally ask 20 questions per passage.')
        st.write('- Some text passages are not suited for 20 questions. Do not make up very constructed and complicated questions just to fill up the 20 - move on to the next text.')
        st.write('- Try to ask questions covering the whole passage and focus on questions covering important information. Do not only ask questions about a single sentence in that passage')
    good_span_expander = st.beta_expander("What is a good answer span?", expanded=False)
    with good_span_expander:
        st.write('- Always mark whole words. Do not start or end the answer within a word.')
        st.write('- For short answers: The answer should be as short and as close to a spoken human answer as possible. Do not include punctuation.')
        st.write('- For long answers: Please mark whole sentences with punctuation. The sentences can also pick up parts of the question, or mark even whole text passages. Mark passages only if they are not too large (e.g. not more than 8-10 sentences).')
    long_short_expander = st.beta_expander(" How do I differentiate long vs short answers?", expanded=False)
    with long_short_expander:
        st.write('- If there is a short answer possible you should always select short answer over long answer.')
        st.write('- Short precise answers like numbers or a few words are short answers.')
        st.write('- Long answers include lists of possibilities or multiple sentences are needed to answer the question correctly.')
    multiple_answers_expander = st.beta_expander("How to handle multiple possible answers to a single question?", expanded=False)
    with multiple_answers_expander:
        st.write('- As of now there is no functionality to mark multiple answers per single question.')
        st.write('- Workaround: You can add a question with the same text but different answer selection by using the button below the question list (Button reads “custom question”).')
    grammatically_question_expander = st.beta_expander("What to do with grammatically wrong or incorrectly spelled questions?", expanded=False)
    with grammatically_question_expander:
        st.write('- Include them. When users use the tool and ask questions they will likely contain grammar and spelling errors, too.')
        st.write('- Exception: The question needs to be understandable without reading and interpretation of the corresponding text passage. If you do not understand the question, please mark the question as “I don’t understand the question”.')
    text_passage_expander = st.beta_expander("What to do with text passages that are not properly converted or contain (in part) information that cannot be labelled (e.g. just lists or garbage text)?", expanded=False)
    with text_passage_expander:
        st.write('- Please do not annotate this text')
        st.write('- You can write down what is missing, or the cause why you cannot label the text + the text number and title.')
    if (
        state_question
        and hasattr(state_question, "data_frame")
        and hasattr(state_question, "question_indices")
    ):
        df = state_question.data_frame
        question_indices = state_question.question_indices
    else:
        df = pd.DataFrame({
            'Question': [],
            'Answer1': [],
            'Answer2': [],
            'Answer3': [],
            'Answer4': [],
        })
        question_indices = 0


# Search bar
question = st.text_input("Please provide your query:", value=random_question)
if state_question and state_question.run_query:
    run_query = state_question.run_query
    st.button("Run")
else:
    run_query = st.button("Run")
    state_question.run_query = run_query

raw_json_feedback = ""

# Get results for query
if run_query:
    with st.spinner(
        "Performing neural search on documents... 🧠 \n "
        "Do you want to optimize speed or accuracy? \n"
        "Check out the docs: https://haystack.deepset.ai/docs/latest/optimizationmd "
    ):
        if eval_set_mode and not question in df.values:
            df2 = pd.DataFrame({
                'Question': [question],
                'Answer1': [""],
                'Answer2': [""],
                'Answer3': [""],
                'Answer4': [""],
            })
            df = df.append(df2, ignore_index=True)
            state_question.data_frame = df
            index = df.index
            condition = df["Question"] == question
            question_indices = index[condition]
            state_question.question_indices = question_indices
        results, raw_json = retrieve_doc(question, top_k_reader=top_k_reader, top_k_retriever=top_k_retriever)

    # Show if we use a question of the given set
    if question == random_question and eval_mode:
        st.write("## Correct answers:")
        random_answer

    st.write("## Results:")

    # Make every button key unique
    count = 0
    for result in results:
        if result["answer"]:
            annotate_answer(result["answer"], result["context"])
        else:
            show_plain_documents(result["context"])
        st.write("**Relevance:** ", result["relevance"], "**Source:** ", result["source"])
        if eval_mode or eval_set_mode:
            # Define columns for buttons
            button_col1, button_col2, button_col3, button_col4 = st.beta_columns([1, 1, 1, 6])
            if button_col1.button("👍", key=(result["context"] + str(count))):
                if eval_mode:
                    raw_json_feedback = feedback_doc(
                        question, "true", result["document_id"], 1, "true", result["answer"], result["offset_start_in_doc"]
                    )
                    st.success("Thanks for your feedback")
                else:
                    if df["Answer1"].values[question_indices] == ['']:
                        df["Answer1"][question_indices] = result["answer"]
                    elif df["Answer2"].values[question_indices] == ['']:
                        df["Answer2"][question_indices] = result["answer"]
                    elif df["Answer3"].values[question_indices] == ['']:
                        df["Answer3"][question_indices] = result["answer"]
                    elif df["Answer4"].values[question_indices] == ['']:
                        df["Answer4"][question_indices] = result["answer"]
            if button_col2.button("👎", key=(result["context"] + str(count))):
                raw_json_feedback = feedback_doc(
                    question,
                    "false",
                    result["document_id"],
                    1,
                    "false",
                    result["answer"],
                    result["offset_start_in_doc"],
                )
                st.success("Thanks for your feedback!")
            if button_col3.button("👎👍", key=(result["context"] + str(count))):
                raw_json_feedback = feedback_doc(
                    question, "false", result["document_id"], 1, "true", result["answer"], result["offset_start_in_doc"]
                )
                st.success("Thanks for your feedback!")
            count += 1
        st.write("___")
    if debug:
        st.subheader("REST API JSON response")
        st.write(raw_json)

if eval_set_mode:
    remove_row = st.text_input("Remove row from table:", value="")
    remove_button = st.button("Remove")
    if remove_button:
        df = df.drop([int(remove_row)], axis=0)
        state_question.data_frame = df
    st.table(df)