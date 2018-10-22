# attack-randomized-models
This is the repo for attack randomized models

## Data
+ CIFAR10

## Network
+ VGG16 `Plain`: No defense
+ VGG16 `RSE`: Random Self-ensemble
+ VGG16 `Adv`: Adversarial training
+ VGG16 `Adv_vi`: Adversarial training Bayesian neural network

## Howto

### Run attacker
```
	python main.py --attack "NES" --model "rse" --steps 100 --eps 0.2
```

### Help
```
	python main.py --help
``` 
