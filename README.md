# Mango Split Dataset
## How to run 
### Inputs
- input source dir of the input imgs
- output source dir of the split imgs
- no. of output imgs desired (10,000)
```bash
python split-v2.py --source "C:\input" --output "E:\output" --massive-augment 10000
```
## Example CLI INPUT
```bash
python split-v2.py --source "C:\Users\Kenan\Downloads\Dataset_mangoes-20250804T050733Z-1-002\Dataset_mangoes\sorted\mango_dataset_v3\sorted_1_1" --output "E:\trash" --massive-augment 10000
```

## Example CLI OUTPUT
```bash
Class Mapping:
------------------------------
green        -> ripeness/green
yellow       -> ripeness/yellow
yellow_green -> ripeness/yellow_green
bruised      -> bruises/bruised
unbruised    -> bruises/not_bruised
Splitting dataset into hierarchical structure...
Processing green -> ripeness/green
  Train: 1225, Val: 262, Test: 263
Processing yellow -> ripeness/yellow
  Train: 616, Val: 132, Test: 132
Processing yellow_green -> ripeness/yellow_green
  Train: 935, Val: 200, Test: 201
Processing bruised -> bruises/bruised
  Train: 1363, Val: 292, Test: 293
Processing unbruised -> bruises/not_bruised
  Train: 1143, Val: 245, Test: 246
Applying massive augmentation to generate 10000 additional images...
Total augmentation combinations available: 309
Original training images: 6832
Target augmentations per original image: 2
Massively augmenting ripeness category...
  Augmenting class: ripeness/green
    Progress: 3.2% (50/1566)
    Progress: 6.4% (100/1566)
    Progress: 9.6% (150/1566)
    Progress: 12.8% (200/1566)
    Progress: 16.0% (250/1566)
    Progress: 19.2% (300/1566)
    Progress: 22.3% (350/1566)
    Progress: 25.5% (400/1566)
    Progress: 28.7% (450/1566)
    Progress: 31.9% (500/1566)
    Progress: 35.1% (550/1566)
    Progress: 38.3% (600/1566)
    Progress: 41.5% (650/1566)
    Progress: 44.7% (700/1566)
    Progress: 47.9% (750/1566)
    Progress: 3.2% (50/1566)
    Progress: 6.4% (100/1566)
    Progress: 9.6% (150/1566)
    Progress: 12.8% (200/1566)
    Progress: 16.0% (250/1566)
    Progress: 19.2% (300/1566)
    Progress: 22.3% (350/1566)
    Progress: 25.5% (400/1566)
    Progress: 28.7% (450/1566)
    Progress: 31.9% (500/1566)
    Progress: 35.1% (550/1566)
    Progress: 38.3% (600/1566)
    Progress: 41.5% (650/1566)
    Progress: 44.7% (700/1566)
    Progress: 47.9% (750/1566)
    Added 3132 augmented images to green
  Augmenting class: ripeness/yellow
    Progress: 6.2% (50/802)
    Progress: 12.5% (100/802)
    Progress: 18.7% (150/802)
    Progress: 24.9% (200/802)
    Progress: 31.2% (250/802)
    Progress: 37.4% (300/802)
    Progress: 43.6% (350/802)
    Progress: 49.9% (400/802)
    Progress: 6.2% (50/802)
    Progress: 12.5% (100/802)
    Progress: 18.7% (150/802)
    Progress: 24.9% (200/802)
    Progress: 31.2% (250/802)
    Progress: 37.4% (300/802)
    Progress: 43.6% (350/802)
    Progress: 49.9% (400/802)
    Added 1604 augmented images to yellow
  Augmenting class: ripeness/yellow_green
    Progress: 4.1% (50/1226)
    Progress: 8.2% (100/1226)
    Progress: 12.2% (150/1226)
    Progress: 16.3% (200/1226)
    Progress: 20.4% (250/1226)
    Progress: 24.5% (300/1226)
    Progress: 28.5% (350/1226)
    Progress: 32.6% (400/1226)
    Progress: 36.7% (450/1226)
    Progress: 40.8% (500/1226)
    Progress: 44.9% (550/1226)
    Progress: 48.9% (600/1226)
    Progress: 4.1% (50/1226)
    Progress: 8.2% (100/1226)
    Progress: 12.2% (150/1226)
    Progress: 16.3% (200/1226)
    Progress: 20.4% (250/1226)
    Progress: 24.5% (300/1226)
    Progress: 28.5% (350/1226)
    Progress: 32.6% (400/1226)
    Progress: 36.7% (450/1226)
    Progress: 40.8% (500/1226)
    Progress: 44.9% (550/1226)
    Progress: 48.9% (600/1226)
    Added 2452 augmented images to yellow_green
Massively augmenting bruises category...
  Augmenting class: bruises/bruised
    Progress: 2.8% (50/1764)
    Progress: 5.7% (100/1764)
    Progress: 8.5% (150/1764)
    Progress: 11.3% (200/1764)
    Progress: 14.2% (250/1764)
    Progress: 17.0% (300/1764)
    Progress: 19.8% (350/1764)
    Progress: 22.7% (400/1764)
    Progress: 25.5% (450/1764)
    Progress: 28.3% (500/1764)
    Progress: 31.2% (550/1764)
    Progress: 34.0% (600/1764)
    Progress: 36.8% (650/1764)
    Progress: 39.7% (700/1764)
    Progress: 42.5% (750/1764)
    Progress: 45.4% (800/1764)
    Progress: 48.2% (850/1764)
    Progress: 2.8% (50/1764)
    Progress: 5.7% (100/1764)
    Progress: 8.5% (150/1764)
    Progress: 11.3% (200/1764)
    Progress: 14.2% (250/1764)
    Progress: 17.0% (300/1764)
    Progress: 19.8% (350/1764)
    Progress: 22.7% (400/1764)
    Progress: 25.5% (450/1764)
    Progress: 28.3% (500/1764)
    Progress: 31.2% (550/1764)
    Progress: 34.0% (600/1764)
    Progress: 36.8% (650/1764)
    Progress: 39.7% (700/1764)
    Progress: 42.5% (750/1764)
    Progress: 45.4% (800/1764)
    Progress: 48.2% (850/1764)
    Added 3528 augmented images to bruised
  Augmenting class: bruises/not_bruised
    Progress: 3.4% (50/1474)
    Progress: 6.8% (100/1474)
    Progress: 10.2% (150/1474)
    Progress: 13.6% (200/1474)
    Progress: 17.0% (250/1474)
    Progress: 20.4% (300/1474)
    Progress: 23.7% (350/1474)
    Progress: 27.1% (400/1474)
    Progress: 30.5% (450/1474)
    Progress: 33.9% (500/1474)
    Progress: 37.3% (550/1474)
    Progress: 40.7% (600/1474)
    Progress: 44.1% (650/1474)
    Progress: 47.5% (700/1474)
    Progress: 3.4% (50/1474)
    Progress: 6.8% (100/1474)
    Progress: 10.2% (150/1474)
    Progress: 13.6% (200/1474)
    Progress: 17.0% (250/1474)
    Progress: 20.4% (300/1474)
    Progress: 23.7% (350/1474)
    Progress: 27.1% (400/1474)
    Progress: 30.5% (450/1474)
    Progress: 33.9% (500/1474)
    Progress: 37.3% (550/1474)
    Progress: 40.7% (600/1474)
    Progress: 44.1% (650/1474)
    Progress: 47.5% (700/1474)
    Added 2948 augmented images to not_bruised
Total augmented images created: 13664
Target was: 10000

Dataset Statistics:
============================================================

RIPENESS Category:
----------------------------------------
  green        - Train: 7830, Val:  488, Test:  478
  yellow       - Train: 4010, Val:  242, Test:  248
  yellow_green - Train: 6130, Val:  376, Test:  376
  Subtotal     - Train: 17970, Val: 1106, Test: 1102

BRUISES Category:
----------------------------------------
  bruised      - Train: 8820, Val:  526, Test:  538
  not_bruised  - Train: 7370, Val:  446, Test:  450
  Subtotal     - Train: 16190, Val:  972, Test:  988

============================================================
TOTAL        - Train: 34160, Val: 2078, Test: 2090
Ratios       - Train: 89.1%, Val: 5.4%, Test: 5.5%

Dataset processing complete! Output saved to: E:\trash

Output Directory Structure:
========================================

dataset_split/
├── train/
│   ├── ripeness/
│   │   ├── green/
│   │   ├── yellow/
│   │   └── yellow_green/
│   └── bruises/
│       ├── bruised/
│       └── not_bruised/
├── val/
│   ├── ripeness/
│   │   ├── green/
│   │   ├── yellow/
│   │   └── yellow_green/
│   └── bruises/
│       ├── bruised/
│       └── not_bruised/
└── test/
    ├── ripeness/
    │   ├── green/
    │   ├── yellow/
    │   └── yellow_green/
    └── bruises/
        ├── bruised/
        └── not_bruised/
```
