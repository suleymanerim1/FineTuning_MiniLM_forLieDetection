# LLM Lie Detection - Fine-Tuning MiniLM for Lie Detection Task

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Humans are not very good at telling when someone is lying  and human accuracy in detecting deception is very constrained. <br>
This study delves into the fine-tuning of MiniLM Language Model, exploring the efficacy in discerning deceit and lie detection.

Project By : <br>
□ Cveevska Marija <br>
□ Erim Suleyman <br>
□ Karakus Isikay <br>
□ Varagnolo Mattia 


## About
This repository encompasses the code and documentation for the "Fine Tuning of MiniLM for Lie Detection Task" project, conducted as part of the Cognitive, Behavioral, and Social Data course at the University of Padua. The project delves into the utilization of Large Language Models (LLMs), particularly MiniLM, for lie detection, employing two distinct methods: transfer learning and parameter-efficient fine-tuning with LoRa.



## Getting Started
Follow these instructions to set up and run the project locally. Ensure you have the necessary dependencies installed.



### Prerequisites
- Python (>=3.6)
- CUDA Environment 



### Required Python Packages
Ensure you have the following Python packages installed:

- **pandas** (>=0.25.0)
- **numpy** (>=1.18.0)
- **torch** (>=1.0.0)
- **sentencepiece** (>=0.1.0)
- **transformers** (>=4.0.0)
- **datasets** (>=1.1.0)
- **peft** (>=1.0.0)
- **wandb** (>=0.10.0)
- **scikit-learn** (>=0.22.0)


### Installation
1. Clone the repository: `git clone https://github.com/marijacveevska/fine-tuning_LLM_for_lie_detection`
2. Change into the project directory: `cd fine-tuning_LLM_for_lie_detection`
3. Ensure you have Python (>=3.6) and a Cuda environment set up.
4. Install dependencies: `pip install -r requirements.txt`


## Project Structure
<details>
<summary>fine-tuning_LLM_for_lie_detection/</summary>
  <ul>
    <li><details>
      <summary>data/</summary>
        <ul>
          <li>Dataset.csv</li>
        </ul>
    </details></li>
    <li><details>
      <summary>notebooks/</summary>
        <ul>
          <li>Transfer_learning.ipynb</li>
        </ul>
    </details></li>
    <li><details>
      <summary>src/</summary>
        <ul>
          <li>load_model.py</li>
          <li>main_LoRA.py</li>
          <li>train.py</li>
        </ul>
    </details></li>
    <li>Fine_Tuning_of_MiniLM_Fin_report.pdf</li>
    <li>MiniLM_Fine_tuning_presentation.pdf</li>
    <li>README.md</li>
    <li>requirements.txt</li>
  </ul>
</details>



## Files
- **Dataset.csv:** Dataset file.
- **Fine_Tuning_of_MiniLM_Fin_report.pdf:** Final project report in PDF format.
- **MiniLM_Fine_tuning_presentation.pdf:** Presentation slides in PDF format.
- **Transfer_learning.ipynb:** Jupyter notebook for transfer learning.
- **load_model.py:** Python script for loading models.
- **main_LoRA.py:** Python script with LoRA implementation.
- **train.py:** Python script for training.

## Results 

Our project yielded compelling results in the lie-detection task, treating it as a binary classification challenge. Through the fine-tuning of MiniLM, we achieved an impressive maximum accuracy of 70%, surpassing human performance in discerning truthful and deceptive instances from raw texts. This outcome underscores the efficacy of Transfer Learning and demonstrates the potential for advanced language models to excel in intricate classification tasks. 
