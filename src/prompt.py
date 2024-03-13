prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Generate a cooking recipe using the following ingredients:
Ingredients: {question}

Only return the helpful recipe below and nothing else.
Helpful recipe:
"""
