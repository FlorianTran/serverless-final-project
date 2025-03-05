# serverless-final-project
## Init
(python 3.10.12)
- python3 -m venv env
- source env/bin/activate
- pip install -r requirements.txt

Inside :
"/venv/lib/python3.10/site-packages/inferrvc/pipeline.py"

change 
```python
_gpu = torch.device("cuda:0")
```
to
```python
_gpu = torch.device("cpu")
```

`serverless s3 start`

Puis dans un autre terminal

`npm run dev`

## WIP ^^^^^^^^^^^^^^^^^^^^^

pour tester le script de creation de tts avec modele d'ia
```bash
python script.py --text "bonjour le monde, est-ce que vous aller bien"
```