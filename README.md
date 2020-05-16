# Face-Mask-Detection

Right now, the world is suffering from COVID-19 pandemic and every day thousand's of people are dying. So to help them, I decided to make software which will allow only those people who are wearing masks to ensure safety. Works in real-time on webcam feed.

If you want to use this, you don't need to train the model (Though training time was really less even on CPU). Model weights are already provided. My best result was with 0.02 loss and 0.022 val_loss and 0.9972 accuracy. So just directly run detect.py.

correct_filenames.py was used to organize dataset in a proper manner.
train_model.py was used to preprocess the data set, build and train model.
MobileNetV2 was used for preprocessing.
