from setuptools import setup, find_packages

setup(
    name="capstone_rag_project",
    version="0.1.0",
    description="A Retrieval-Augmented Generation system for UQ policy documents",
    author="Noah Smyth",
    url="https://github.com/KnowerSmyf/rag-system",
    packages=find_packages(),
    install_requires=[
        "dspy",               # Updated package name for DSPy
        "spacy",
        "chromadb",
        "langchain",
        "sentence-transformers",
        "torch",
        "python-dotenv",
        "pandas",
        "seaborn",
        "matplotlib"
    ]
)