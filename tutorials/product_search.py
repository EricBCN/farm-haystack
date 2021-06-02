import pandas as pd
from pathlib import Path
from haystack.utils import launch_es
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.reader.farm import FARMReader
from haystack.pipeline import ExtractiveQAPipeline
from haystack import Label
from haystack.pipeline import Pipeline
from haystack.eval import EvalDocuments


data = Path("../data/SubjQA/SubjQA")

# Taken straight from book draft
category = "electronics"
dfs = {p.stem: pd.read_csv(p) for p in (data/category).glob("**/*.csv")}
for split, df in dfs.items():
    print(f"Number of questions in {split}: {df['q_review_id'].nunique()}")

# BC:
# Utility function to make it easier to start ES
# Still requires docker in order to work
launch_es()

document_store = ElasticsearchDocumentStore(return_embedding=True)
document_store.delete_documents(index="document")
document_store.delete_documents(index="label")


for split, df in dfs.items():
    # Exclude duplicate reviews and slice off ANSWERNOTFOUND from each review
    docs = [
        {
            "text": row["review"][:-15],
            "meta":{
                "item_id": row["item_id"],
                "qid": row["q_review_id"],
                "split": split
            }
        }
        for _, row in df.drop_duplicates(subset="review_id").iterrows()
    ]
    document_store.write_documents(docs, index="document")

es_retriever = ElasticsearchRetriever(document_store=document_store)

max_seq_length = 384
doc_stride = 128
reader = FARMReader(
    model_name_or_path="deepset/roberta-base-squad2",
    progress_bar=False,
    max_seq_len=max_seq_length,
    doc_stride=doc_stride,
    return_no_answer=True)

# BC: EvalRetriever has been renamed to EvalDocuments() since this can also be used for the reranker
# BC: EvalReader has also been renamed EvalAnswers()
class EvalRetrieverPipeline:
    def __init__(self, retriever):
        self.retriever = retriever
        self.eval_retriever = EvalDocuments()
        pipe = Pipeline(pipeline_type="Query")
        pipe.add_node(component=self.retriever, name="ESRetriever", inputs=["Query"])
        pipe.add_node(component=self.eval_retriever, name="EvalRetriever", inputs=["ESRetriever"])
        self.pipeline = pipe

pipe = EvalRetrieverPipeline(es_retriever)

labels = []
# Get list of unique question IDs
qids = dfs["test"]["q_review_id"].unique()
for qid in qids:
    # Collect all answers associated with a question ID
    df = dfs["test"].query(f"q_review_id == '{qid}'")
    # Iterate over each question-answer pair and add to labels
    for _, row in df.iterrows():
        # Unanswerable questions are empty strings in SQuAD format
        answer = (row["human_ans_spans"] if row["human_ans_spans"] != "ANSWERNOTFOUND" else "")
        label = Label(
            question=row["question"],
            answer=answer,
            id=qid,
            origin="SubjQA",
            meta={"item_id": row["item_id"]},
            is_correct_answer=True,
            is_correct_document=True,
            no_answer=True if answer == "" else False,
        )
        labels.append(label)

# BC: You can use get_label_count instead of len(document_store.get_all_documents())
document_store.write_labels(labels, index="label")
print("n_labels in document store")
print(document_store.get_label_count(index="label"))

# BC: This simplifies the loop that you had to get the labels
labels_agg = document_store.get_all_labels_aggregated(
    index="label",
    aggregate_by_meta=["item_id"]
)
print("n_aggregated_labels")
print(len(labels_agg))


#BC: Is it right that we go from 358 labels (unaggregated) to 330?


#BC: no longer need to insert label objects into a dict, can get rid of the qid2label variable
#BC: query comes from Label.question and meta filter populated by Label.meta
def run_pipeline(pipeline, top_k_retriever=10, top_k_reader=4):
    for l in labels_agg:
        _ = pipeline.pipeline.run(
            query=l.question,
            top_k_retriever=top_k_retriever,
            top_k_reader=top_k_reader,
            top_k_eval_documents=top_k_retriever,    # This is to set the top_k of the EvalDocuments node
            labels=l,
            filters={
                "item_id": [l.meta["item_id"]],
                "split": ["test"]
            }
        )

# BC: We now calculate MRR as well as Recall!
run_pipeline(pipe, top_k_retriever=3)
pipe.eval_retriever.print()

print()
