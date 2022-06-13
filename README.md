# STEPS_Bioinformatics
we propose a novel **ST**rucure-awar**E** **P**rotein **S**elf-supervised Learning (**STEPS**) method.

## Pretraining
```bash
python -u main.py --mode prt
```

## Finetuning
```bash
python -u main.py --mode ft --task water

python -u main.py --mode ft --task loc

python -u main.py --mode ft --task enzyme
```
