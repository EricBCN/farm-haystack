import random
import uuid

from itertools import chain, islice

from haystack import Label
from haystack.nodes.base import BaseComponent
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import QuestionGenerator, SentenceTransformersRanker, BaseRetriever


# batching a generator without pre-walking
# https://stackoverflow.com/questions/24527006/split-a-generator-into-chunks-without-pre-walking-it/24527424#24527424
def chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


class PseudoLabelGenerator(BaseComponent):
    """
    Node that is used to generate pseudo-labeled data for unsupervised domain adaptation of dense retriever models.
    See Wang et al. https://arxiv.org/pdf/2112.07577.pdf
    """
    outgoing_edges = 1

    def __init__(
            self,
            document_store: ElasticsearchDocumentStore,
            retriever: BaseRetriever,
            question_generator: QuestionGenerator,
            ranker: SentenceTransformersRanker,
            doc_index: str = 'docs',
            target_index: str = 'gpl',
            batch_size: int = 50,
            num_negatives: int = 50
    ):
        """
        :param document_store: Instance of DocumentStore where the documents for unsupervised label gen are indexed.
        :param retriever: Instance of Retriever that will be used to mine hard negatives.
        :param question_generator: Instance of QuestionGenerator that will be used to generate questions.
        :param ranker: Instance of Ranker that will be used to predict a similarity score between query<>positive
            and query<>negative
        :param doc_index: name of the index where the indexed documents can be queried.
        :param target_index: name of the index where the PseudoLabelGenerator should store generated labels
        :param batch_size: how many documents should be used per question generation run.
        :param num_negatives: How many negatives should be mined.
        """
        self.set_config(
            document_store=document_store,
            retriever=retriever,
            question_generator=question_generator,
            ranker=ranker,
            doc_index=doc_index,
            target_index=target_index,
            batch_size=batch_size,
            num_negatives=num_negatives
        )

        self.document_store = document_store
        self.retriever = retriever
        self.ranker = ranker
        self.question_generator = question_generator
        self.doc_index = doc_index
        self.target_index = target_index
        self.batch_size = batch_size
        self.num_negatives = num_negatives

    def generate_questions(self):
        documents = self.document_store.get_all_documents_generator(index=self.doc_index)
        batches = chunks(documents, self.batch_size)

        for batch in batches:
            docs = list(batch)
            generated, _ = self.question_generator.run(docs)
            questions = [q['questions'] for q in generated['generated_questions']]
            doc_output = generated['documents']
            self._write_generated_questions(questions, doc_output)

    def mine_negatives(self):
        # this should be replaced by a generator version -> see branch for label generator
        labels = self.document_store.get_all_labels(index=self.target_index)

        for label in labels:
            results = self.retriever.retrieve(query=label.query, top_k=self.num_negatives, index=self.doc_index)
            valid_negatives = []
            for result in results:
                if result.id != label.document.id:
                    negative = Label(
                        query=label.query,
                        answer=None,
                        is_correct_document=False,
                        is_correct_answer=False,
                        document=result,
                        origin='user-feedback',
                        meta={'query_id': label.meta['query_id']}
                    )
                    valid_negatives.append(negative)
            self.document_store.write_labels(labels=valid_negatives, index=self.target_index)

    def generate_scored_samples(self):
        multilabels = self.document_store.get_all_labels_aggregated(
            index=self.target_index,
            open_domain=True,
            drop_negative_labels=False,
            drop_no_answers=False,
            aggregate_by_meta='query_id'
        )

        quartets = []
        for multilabel in multilabels:
            query = multilabel.query
            negative_documents = [label.document for label in multilabel.labels if not label.is_correct_document]
            positive_document = [label.document for label in multilabel.labels if label.is_correct_document]
            if len(positive_document) > 0 and len(negative_documents) > 0:
                negative_document = negative_documents[random.randrange(len(negative_documents))]
                predicted_documents = self.ranker.predict(query, [positive_document[0], negative_document], top_k=2)

                scores = [doc.score for doc in predicted_documents]
                score = scores[0] - scores[1]

                quartets.append((query, positive_document[0].content, negative_document.content, score))

        return quartets

    def run(self):
        self.generate_questions()
        self.mine_negatives()
        samples = self.generate_scored_samples()

        return {'samples': samples}, 'output_1'

    def _write_generated_questions(self, questions, documents):
        labels = []

        for idx, doc in enumerate(documents):
            question_sample = questions[idx]
            for question in question_sample:
                label = Label(
                    query=question,
                    document=doc,
                    answer=None,
                    is_correct_answer=False,
                    is_correct_document=True,
                    origin='user-feedback',
                    meta={'query_id': uuid.uuid4()}
                )
                labels.append(label)

        self.document_store.write_labels(labels=labels, index=self.target_index)
