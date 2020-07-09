# Semantic segmentation demo

This is a simple iOS app that uses the **DeepLab V3+** semantic segmentation model on the real-time camera feed and applies some graphical effects to the image using the predicted segmentation mask.

THIS IS NOT PRODUCTION-QUALITY CODE! But you might find it useful as a demo of how to take an `MLMultiArray` output from Core ML and draw it on the screen using Metal.

Note that the DeepLab model applies an argmax to the output and returns the segmentation mask as one INT32 value for each pixel. For more info about this, see my e-book [Core ML Survival Guide](http://leanpub.com/coreml-survival-guide).

## Usage instructions:

- tap camera button to switch between front and back camera

- use menu to choose a graphical effect

- drag controller (white circle) up/down and left/right to change settings for current graphical effect

- tap bottom left corner to enable/disable live video preview (useful for debugging)

## Available graphical effects

- blend between original input and segmentation mask (all classes)

- draw shadow on a fake background; controller determines position of the "light source"

- change brightness and saturation (using RGB <-> HSV conversion)

- pixelate the background and/or the person

- blur the background (the blur size is fixed)

- add a simple purple-ish glow around any persons

## License

Do whatever you want with it. I DO NOT OFFER ANY SUPPORT FOR THIS REPO!

The DeepLab model is taken from [TensorFlow](https://github.com/tensorflow/models/tree/master/research/deeplab) and is licensed under the Apache License.
