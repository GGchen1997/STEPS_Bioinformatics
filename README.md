# STEPS_NMI


### Pretraining
python -u main.py --mode prt --use_lm 1 --prt_coeff 0.05 --prt_trade global --base_model bert

### Finetuning
python -u main.py --mode ft --ft_mode bilevel-b --task water  --use_lm 1 --prt_coeff 0.05 --prt_trade global --base_model bert

python -u main.py --mode ft --ft_mode bilevel-b --task loc  --use_lm 1 --prt_coeff 0.05 --prt_trade global --base_model bert

python -u main.py --mode ft --ft_mode bilevel-b --task enzyme  --use_lm 1 --prt_coeff 0.05 --prt_trade global --base_model bert
