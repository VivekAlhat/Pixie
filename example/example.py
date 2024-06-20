import os
import sys
import warnings

warnings.filterwarnings("ignore")

import ollama
import numpy as np
from sentence_transformers import SentenceTransformer

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

from pixie import Pixie


# creating an instance of embedder model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# creating an instance of pixie vector store
pixie = Pixie(embedder)


# generate an answer using llama3 and context docs
def generate_answer(prompt):
    response = ollama.chat(
        model="llama3",
        options={"temperature": 0.7},
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return response["message"]["content"]


with open("example/spacebattle.txt") as f:
    content = f.read()
    ingested = pixie.from_docs(docs=content.split("\n\n"))
    print(ingested)

# system prompt
PROMPT = """
    User has asked you following question and you need to answer it based on the below provided context.
    If you don't find any answer in the given context then just say 'I don't have answer for that'.
    In the final answer, do not add "according to the context or as per the context".
    You can be creative while using the context to generate the final answer. DO NOT just share the context as it is.

    CONTEXT: {0}
    QUESTION: {1}

    ANSWER HERE:
"""

while True:
    query = input("\nAsk anything: ")
    if len(query) == 0:
        print("Ask a question to continue...")
        quit()

    if query == "/bye":
        quit()

    similarities = pixie.similarity_search(query, top_k=5)
    print(f"query: {query}, top {len(similarities)} matched results:\n")

    print("-" * 5, "Matched Documents Start", "-" * 5)
    for match in similarities:
        print(f"{match}\n")
    print("-" * 5, "Matched Documents End", "-" * 5)

    context = ",".join(similarities)
    answer = generate_answer(prompt=PROMPT.format(context, query))
    print("\n\nQuestion: {0}\nAnswer: {1}".format(query, answer))

    continue
