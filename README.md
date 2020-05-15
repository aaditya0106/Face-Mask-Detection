# Face-Mask-Detection

Real time face mask detection on webcam.

If you want to use this, you don't need to train the model (Though training time was really less even on CPU). Model weights are already provided. My best result was with 0.02 loss and 0.022 val_loss and 0.9972 accuracy. So just directly run detect.py.

correct_filenames.py was used to organize dataset in a proper manner.
train_model.py was used to preprocess the data set, build and train model.
MobileNetV2 was used for preprocessing.
