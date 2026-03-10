# Agentic RAG System for Policy Search

## Overview
This project investigates retrieval-augmented generation (RAG) for question answering over the University of Queensland Policy and Procedure Library (PPL), a compliance-critical document domain where retrieval quality and hallucination mitigation are especially important.

The system compares multiple retrieval design choices, including:
- chunking strategy (agentic vs text-split)
- retrieval architecture (ReAct vs retrieve-rerank)
- DSPy prompt compilation / optimisation

## Motivation
The existing PPL search experience is primarily keyword-based, requiring users to manually inspect many documents. This project explores whether a RAG-based system can provide more direct, grounded answers while preserving reliability in a high-stakes domain.

## System Architecture
The project consists of five main stages:
1. Scraping and metadata extraction from the PPL website
2. Document preprocessing and corpus construction
3. Index construction for multiple retrieval variants
4. Retrieval and answer generation
5. Synthetic QA generation and evaluation

## Experimental Variants
### Chunking
- Text-split chunking baseline
- Agentic / structure-aware chunking

### Retrieval
- Traditional retrieve-rerank pipeline
- ReAct-style agentic retrieval pipeline

### Optimisation
- DSPy compilation for prompt / program optimisation

## Repository Structure
The repository is organised around a modular pipeline that separates data ingestion, indexing, retrieval architectures, and evaluation experiments.

```text
src/
  scraping/           # HTML scraping, metadata extraction, preprocessing
  data_processing/    # indexing, dataset generation, caching
  rag_system/         # retrieval architectures, tools, data stores, DSPy signatures
  evaluation/         # evaluation metrics and experiment runner

tasks/
  scraping/           # end-to-end scraping scripts
  data_preparation/   # build vector stores, metadata stores, and QA datasets
  training/           # DSPy program compilation
  evaluation/         # run evaluation experiments

notebooks/
  agentic_rag.ipynb           # walkthrough of the agentic RAG pipeline and example system outputs
  evaluation_analysis.ipynb   # result analysis, summary statistics, and figures
  figures/                    # plots generated during evaluation

results/
  final_evaluation_summary.csv
  final_evaluation_detailed.csv
  optimized_programs/         # compiled DSPy retrieval programs

data/
  scraped_corpus.jsonl
  stores/                     # vector stores and metadata indices
  synthetic_dataset/          # generated QA datasets used for evaluation
```

## Data Pipeline

- Raw policy HTML and metadata are scraped from the UQ PPL

- Documents are processed into a structured corpus format using `Html2Text`

- Multiple retrieval stores are built for different chunking / indexing strategies

- Synthetic single-hop and multi-hop QA datasets are generated for evaluation

## Evaluation

The project includes:

- synthetic train/test QA datasets

- controlled ablation experiments

- summary and detailed evaluation outputs

- analysis notebooks and figures

## Key Outputs

`results/final_evaluation_summary.csv`

`results/final_evaluation_detailed.csv`

`notebooks/evaluation_analysis.ipynb`

`notebooks/figures/`

## Notes

- Large datasets and vector stores are not fully included in the repository due to size constraints.
- This repository contains the project code, notebooks, and selected outputs, but does not include all data artefacts and vector stores required for full reproduction due to size constraints. 

## Lessons / Future Work

This project highlighted the importance of:

- preserving document structure during chunking

- carefully evaluating retrieval metrics

- testing design assumptions empirically rather than relying on intuition alone

