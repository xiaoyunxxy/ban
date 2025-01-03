# BAN: Detecting Backdoors Activated by Adversarial Neuron Noise

## Environment settings
check requirements.txt

## Detection

```
CUDA_VISIBLE_DEVICES=0 python ban_detection.py --checkpoint ${ckpt} --arch resnet18 --mask-lambda 0.75 --steps 30 --eps 0.3 --print-every 20
```

## Mitigation

```
CUDA_VISIBLE_DEVICES=0 python ban_finetune.py --epoch 25 --attack_mode all2one --lr 0.005 --rob-lambda 0.5 --checkpoint ${ckpt} --poison-type bpp --trigger-alpha 1.0
```

## Pretrained example models

You can download the trained backdoor models: [link](https://drive.google.com/drive/folders/164Auz6sk-ieadIUZ43E26e6VnbLYk-X6?usp=sharing)
