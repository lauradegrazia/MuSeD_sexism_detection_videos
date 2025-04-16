# MuSeD: A Multimodal Spanish Dataset for Sexism Detection

This repository is associated with the research paper titled [**MuSeD: A Multimodal Spanish Dataset for Sexism Detection in Social Media Videos**](https://arxiv.org/abs/2504.11169)

*MuSeD: A Multimodal Spanish Dataset for Sexism Detection in Social Media Videos* Laura De Grazia, Pol Pastells, Mauro Vázquez Chas, Desmond Elliott, Danae Sánchez Villegas, Mireia Farrús, Mariona Taulé. 

MuSeD is a Multimodal Spanish dataset for Sexism Detection on social media videos. 

Content is considered sexist in four main cases:
* Stereotype: It attributes a set of properties that supposedly differentiate men and women, based on stereotypical beliefs;
* Inequality: It asserts that gender inequalities no longer exist and that the feminist movement is marginalizing the position of men in society;
* Discrimination: It discriminates against the LGBTQ+ community;
* Objectification: It portrays women as physical objects, often hypersexualizing their bodies. 

The following figure illustrates a sexist video that includes a gender stereotype:

<img width="255" alt="stereotype" src="https://github.com/user-attachments/assets/289dbe15-ea76-470e-a331-5309eb1c6b6c" />

The dataset will be publicly available for research purposes.

## Human annotation
MuSeD was annotated by a team of six annotators from diverse gender and age backgrounds to mitigate demographic bias. In our annotation process, the annotators classified content as sexist or non-sexist across different modalities. The process consisted of three stages: first, annotators labeled the transcript of the audio and OCR texts; second, they annotated the audio; and finally, they annotated the entire video, which included all modalities.

## Dataset statistics 
MuSeD includes 400 videos, ≈ 11 hours, extracted from two different platforms: TikTok, a moderated platform, and BitChute, a low-moderation platform, which enables us to find diverse material. The following figure illustrates the dataset statistics:

<img width="394" alt="Screenshot 2025-04-09 at 15 28 48" src="https://github.com/user-attachments/assets/49cb9f40-a411-47eb-b900-3b31e26645df" />


## Model evaluation 
We evaluate a range of large language models (LLMs) on the task of sexism detection using MuSeD. As in human annotation, the goal is to classify each video as either sexist or non-sexist. Models are evaluated against the text label (assigned by annotators based on the text transcript alone) and the multimodal label (assigned when both text and visual content were available to annotators). The results suggest that larger models generally outperform smaller ones, with the best performance achieved by GPT-4o. The model performance results are as follows:

<img width="419" alt="Screenshot 2025-04-09 at 16 38 48" src="https://github.com/user-attachments/assets/8a1b322f-4857-4a83-879e-0bb8158670da" />


