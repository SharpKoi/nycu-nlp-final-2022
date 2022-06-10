# Empathetic Dialogues Sentiment Analysis

🥳 Thanks to my teammate [Max-0729](https://github.com/Max-0729)

:point_right: Check our [final presentation](https://docs.google.com/presentation/d/149SlUUqYjioZcZIapPJ51buBLMoQRdP4HfOgl9ETHkc/edit?usp=sharing)

## Files Info

| File Name         | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| data/             | the folder storing all fixed data and all pre-processed data |
| data_bugfix.ipynb | the data bug fixing script                                   |
| EDA.ipynb         | the script used to do pre-processing and exploratory data analysis |
| Base Model.ipynb  | the base models training script                              |
| Master.ipynb      | the stacking ensemble model building and training script     |
| transforms.py     | the augmentation tools                                       |

## Usage

To reproduce our result, you need to ensure all the code are placed in the right place. And following the instruction below:

1. You need to download all our base model checkpoints from [here](https://drive.google.com/drive/folders/1QknQHaXQwZHlCPwhWAC3l6OAAap31uTk?usp=sharing). And put the downloaded `models` folder in the project root place.
   - Note that the base models were trained by `Base Model.ipynb`. 
2. Open the `Master.ipynb` and run all the code cells(Google Colab is recommended).
