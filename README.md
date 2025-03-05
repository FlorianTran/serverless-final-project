# serverless-final-project
## Init
(python 3.10.12)
- pyhton3 -m venv env
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