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

# Installation and Running Experiments

## Prerequisites
- Python 3.10.8
- Ollama - required for local LLM inference
- Gemma 4 E4B - The model used for the experiments, downloaded via Ollama
- Git - To clone this repository

## Setup Instructions

1. Install Ollama<br>Download and install Ollama from https://ollama.com
2. Download the required LLM<br>Open a terminal window and pull the required model
   ```bash
   ollama pull gemma4:e4b
   ```
   Ensure Ollama is running before attempting to execute the project
3. Clone the repository
    ```bash
    git clone https://github.com/JoshuaDjordjevic/honours-project/
    cd honours-project
    ```
4. Create a python virtual environment
    ```bash
    python -m venv venv
    ```
5. Activate the virtual environment<br>Windows:
   ```bash
   venv\Scripts\activate
   ```
   Linux / macOS
   ```bash
   source venv/bin/activate
   ```
6. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

### Notes
- Python 3.10.8 is the version used during development and experimentation.
- Ollama manages the local model installation and inference.
- The `gemma4:e4b` model must be available locally before running the experiments. The version used in the experiments was the latest release as of 31/10/2026.
- A virtual environment is recommend to isolate project dependencies.

## Running the Experiment

### Step 1. Prepare the Results Directory

Before starting a new experiment, delete any existing output files in the `results/` directory to prevent previous results from being mixed with new experimental data.

The following files should be removed:

- `results/grading.csv`
- `results/grading.jsonl`
- `results/results.csv`
- `results/results.jsonl`

`results/Results Excel Doc.xlsx` may be kept. This file is an example file containing key data from the thesis experimentation run.

### Step 2. Generate LLM Responses

Run `generate_responses.py` to generate responses from the configured LLM.

The script uses the benchmark questions provided in `benchmark_questions/questions.xlsx` as input and writes the generated responses to `results/results.jsonl`.

If the output file already exists, new results will be appended rather than overwriting existing entries.

### Step 3. Grade the Generated Responses

Once response generation is complete, run `generate_grading.py` to evaluate the generated responses. Grading is based off question specifications provided in `benchmark_questions/questions.xlsx`.

The grading results are saved to `results/grading.jsonl`.

### Step 4. Convert JSONL Files to CSV (Optional)

For easier analysis and visualisation, the JSONL output files can be converted into CSV format.

Run the following scripts:

- `convert_grading.py` - Converts `grading.jsonl` into `grading.csv`.
- `convert_results.py` - Converts `results.jsonl` into `results.csv`.

The resulting CSV files can be opened in Microsoft Excel, Tableau, or other data analysis and visualisation software.

# Research Results

For results of this research please see the thesis.