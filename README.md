<h1 align="center">StoryBench: A Dataset for Diverse, Explainable,Multi-hop Narrative Text-to-Image Generation</h1>
<!-- Clustering and Ranking: Diversity-preserved Instruction Selection through Expert-aligned Quality Estimation -->
<h4 align="center"> Yuan Ge, Kaiyang Ye, Saihan Chen, Aokai Hao, Xiangnan Ma,Kaiyan Chang,Tong Xiao,Jingbo Zhu</h4>

## News💡
- [2025.07] StoryBench is accepted by NLPCC 2025 Oral!🎉🎉🎉

## StoryBench📚 

**StoryBench** is an evaluation dataset for narrative text-to-image (T2I) generation, characterized by its diversity, explainability, and multi-hop nature. It consists of 728 prompts across five categories: animal-nature, labor, medical, sport, and technology. The assessment of five prominent T2I models, including DALL·E 3 and Midjourney, has revealed that even advanced T2I models have limited capability in generating complex event-narrative images.

<p align="center">
    <img src="pic/main.png" width="80%"> <br>
    An overview of StoryBench with 5 categories. 
</p>

## Usage 🛠

### Image generation

Please use the following text-to-image models for image generation and reasoning:

- [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium)

- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

- [FLUX.1 Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)

- [DALL·E 3](https://openai.com/dall-e-3)

- [Midjourney](https://midjourney.com)

- Or any other text-to-image model you prefer  

 > After generating the images, save all output files under the `Image/` directory for further evaluation and analysis.
 > The image saving format follows the example path:`Image/SD3-Zero-hop`

### Evaluation

#### Step 1: Install Visual-Language Models (VLMs)
We use three models, with **InternVL2_5-38B** as the main example. The model is available here:[InternVL2_5-38B](https://huggingface.co/OpenGVLab/InternVL2_5-38B).
    
##### Quick Installation ⚙️:
```bash
conda create -n internvl python=3.9 
conda activate internvl
```
For the rest of the environment setup, please follow the official repository instructions:[InternVL](https://github.com/OpenGVLab/InternVL).
We also use **GPT-4o** and **human evaluation** for comparison.


#### Step 2: Start Evaluation

##### InternVL2\_5-38B Evaluation
- Run the script:
```bash
python code/38B-evaluate.py
```
  > You can adjust the JSON file path and image path as needed.
  > The JSON files are saved under the directory:`data/multi-hop-data`,The images are saved under the directory:`image/`

##### GPT-4o Evaluation
- Run the script:
```bash
python code/4o-evaluate.py
```
 >  Paths can be customized and your own OpenAI API key must be provided.
##### Human Evaluation
  It is recommended to have at least three experienced evaluators score the results based on the scoring criteria.







