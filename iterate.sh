#!/bin/bash
for i in {2..20}
do
    j=$((i-1))
    echo "Processing Stage $i from Previous Stage $j"
    python train_privacy_autoencoder.py --prevstage "$j" --nextstage "$i"
    python train_image_classifier.py --stage "$i"
    python train_code_classifier.py --stage "$i"
done