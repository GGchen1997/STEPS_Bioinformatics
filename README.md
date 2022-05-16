# STEPS_NMI


### Pretraining
python -u main.py --mode prt

### Finetuning
python -u main.py --mode ft --task water

python -u main.py --mode ft --task loc

python -u main.py --mode ft --task enzyme
