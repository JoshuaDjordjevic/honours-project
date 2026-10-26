# Reducing Hallucinations in Small-Scale LLMs using RAG and Callable Tools

## Overview

This project investigates whether Retrieval-Augmented Generation (RAG) and callable external tools can reduce hallucinations and improve the factual reliability of small-scale, open-source Large Language Models (LLMs).

The system is designed to run locally on consumer-grade hardware, providing an accessible alternative to computationally expensive approaches such as fine-tuning and large-scale model retraining.

## Objectives

- Implement a lightweight RAG pipeline to provide relevant external context.
- Integrate callable tools for tasks requiring specialised functionality or real-time information.
- Compare factual accuracy, hallucination frequency, and response quality across different configurations.
- Evaluate inference performance and resource requirements for local deployment.
- Determine whether combining RAG and callable tools provides greater benefits than using either approach independently.

## Experimental Configurations

The project evaluates four LLM configurations:

1. **Baseline:** The model generates responses without retrieval or external tools.
2. **RAG:** The model uses retrieved documents to ground its responses.
3. **Tools:** The model calls external tools to obtain information or perform calculations.
4. **RAG + Tools:** The model combines document retrieval with callable external tools.

## Evaluation

Each configuration is evaluated using consistent prompts, datasets, and inference settings. Evaluation considers factual correctness, hallucination frequency, response quality, inference latency, and computational resource usage.

## Technologies

- **Python** - Core implementation and evaluation.
- **Ollama** - Local LLM inference and prototyping.
- **LangChain** - RAG pipelines and tool integration.
- **Nomic Embed Text** - Document embedding for semantic retrieval.
- **Vector Store** - Storage and retrieval of document embeddings.

## Research Goal

Determine whether RAG and callable tools can improve the factual reliability of small-scale open-source LLMs while maintaining practical performance and accessibility on consumer-grade hardware.