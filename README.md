## Data Analysis Project for CFPB complaint records

**NOTES:** 
1) `results_local/` folder contains the results that were obtained by executing the python program on my machine.
2) `models/` folder contains a shortened version of the model. I used it for faster execution time. The full model can be installed by executing the python script `download_model.py`

### Requirements
`python 3.11.9`

### How to run the project

Execute the following commands (in linux)

```
python3.11 -m venv venv
```
```
source venv/bin/activate
```
```
pip install -r requirements.txt
```
```
python3.11 -m spacy download en_core_web_sm
```
```
NLTK_DATA="./nltk_data" python3.11 process_data.py
```