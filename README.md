# mlart
Machine learning based art using the Keras interface to TensorFlow.

Over-fits a convolutional neural network to the inverse of a downsampling process on a source image:

![image](https://user-images.githubusercontent.com/8690175/235301980-d916998b-c39d-48b0-b64a-93be4021e1ff.png)

The resulting model is then an up-sampler, with the characteristics of the source image imprinted upon it.
One can then apply this upsampler to generate art in the style of the source image:

![image](https://user-images.githubusercontent.com/8690175/235302119-361ecdbd-0c2d-43bf-a20b-ea2c9a6d5041.png)

By applying the model as a stencil onto a target image, up-sampling a series of small chunks of the target 
image and compsiting the result, one can imprint the style of the source image onto the target image:

![image](https://user-images.githubusercontent.com/8690175/235302330-4fcd5c90-7496-4329-ad6b-935ebf0db157.png)

# Usage
To train a model on a single image: `train.py source.jpg` (will produce a model.save)

To train on a set of images: `train.py directory/with/images`

To predict on some pre-determined and randomly-generated inputs: `predict.py model.save`

To predict on the above, as well as a particular target: `predict.py model.save target.jpg`
