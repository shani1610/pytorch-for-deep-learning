# pytorch-for-deep-learning
Follows the course ["Mastery Learn PyTorch for Deep Learning"](https://www.learnpytorch.io/)

| Course Notebook                   | Course Exercises       | What does it cover?                                                                            |
|---------------------------|--------|-----------------------------------------------------------------------------------------------|
| [00 - PyTorch Fundamentals]() | [00 - PyTorch Fundamentals Exe.]() | Many fundamental PyTorch operations used for deep learning and neural networks (Tensor, Operations, Data Manipulation and Indexing, Reproducibility, Run Tensors on GPU).               |
| [01 - PyTorch Workflow]()    | [01 - PyTorch Workflow Exe.]() | Provides an outline for approaching deep learning problems and building neural networks with PyTorch. | 
| [02 - PyTorch Neural Network Classification]()    | [02 - PyTorch Neural Network Classification Exe.]() | Uses the PyTorch workflow from 01 to go through a neural network classification problem. | 
| [03 - PyTorch Computer Vision]()    | [03 - PyTorch Computer Vision Exe.]() | Let's see how PyTorch can be used for computer vision problems using the same workflow from 01 & 02. | 
| [04 - PyTorch Custom Datasets]()    | [04 - PyTorch Custom Datasets Exe.]() | How do you load a custom dataset into PyTorch?  | 


## Main Takeaways:

![01_a_pytorch_workflow](https://github.com/user-attachments/assets/2a43cf03-cc10-409d-b7bf-9523f5cf954e)

<img width="1459" alt="01-pytorch-training-loop-annotated" src="https://github.com/user-attachments/assets/0051b63a-20bb-4657-958b-dd6d92b9cdbc">

<img width="1463" alt="01-pytorch-testing-loop-annotated" src="https://github.com/user-attachments/assets/b635001b-95ac-40be-a960-ca05a649901e">


## Methods to prevent overfitting	

| Method to prevent overfitting                   | What is it?                                                            
|---------------------------|----------------------------------------------------------------------------------------------------| 
| Get more data	| Having more data gives the model more opportunities to learn patterns, patterns which may be more generalizable to new examples. |
|Simplify your model |	If the current model is already overfitting the training data, it may be too complicated of a model. This means it's learning the patterns of the data too well and isn't able to generalize well to unseen data. One way to simplify a model is to reduce the number of layers it uses or to reduce the number of hidden units in each layer.|
| Use data augmentation	| Data augmentation manipulates the training data in a way so that's harder for the model to learn as it artificially adds more variety to the data. If a model is able to learn patterns in augmented data, the model may be able to generalize better to unseen data.|
| Use transfer learning	| Transfer learning involves leveraging the patterns (also called pretrained weights) one model has learned to use as the foundation for your own task. In our case, we could use one computer vision model pretrained on a large variety of images and then tweak it slightly to be more specialized for food images.|
| Use dropout layers	| Dropout layers randomly remove connections between hidden layers in neural networks, effectively simplifying a model but also making the remaining connections better. See torch.nn.Dropout() for more.
| Use learning rate decay	| The idea here is to slowly decrease the learning rate as a model trains. This is akin to reaching for a coin at the back of a couch. The closer you get, the smaller your steps. The same with the learning rate, the closer you get to convergence, the smaller you'll want your weight updates to be.|
| Use early stopping	| Early stopping stops model training before it begins to overfit. As in, say the model's loss has stopped decreasing for the past 10 epochs (this number is arbitrary), you may want to stop the model training here and go with the model weights that had the lowest loss (10 epochs prior).|

## Methods to prevent underfitting	

| Method to prevent underfitting                   | What is it?                                                            
|---------------------------|----------------------------------------------------------------------------------------------------| 
| Add more layers/units to your model	| If your model is underfitting, it may not have enough capability to learn the required patterns/weights/representations of the data to be predictive. One way to add more predictive power to your model is to increase the number of hidden layers/units within those layers.| 
| Tweak the learning rate	| Perhaps your model's learning rate is too high to begin with. And it's trying to update its weights each epoch too much, in turn not learning anything. In this case, you might lower the learning rate and see what happens.| 
| Use transfer learning	| Transfer learning is capable of preventing overfitting and underfitting. It involves using the patterns from a previously working model and adjusting them to your own problem.
| Train for longer	| Sometimes a model just needs more time to learn representations of data. If you find in your smaller experiments your model isn't learning anything, perhaps leaving it train for a more epochs may result in better performance.| 
| Use less regularization	| Perhaps your model is underfitting because you're trying to prevent overfitting too much. Holding back on regularization techniques can help your model fit the data better.| 



## The three big PyTorch errors
Predicting on your own custom data with a trained model is possible, as long as you format the data into a similar format to what the model was trained on. Make sure you take care of the three big PyTorch and deep learning errors:

1) Wrong datatypes - Your model expected torch.float32 when your data is torch.uint8.
  
3) Wrong data shapes - Your model expected [batch_size, color_channels, height, width] when your data is [color_channels, height, width].
  
5) Wrong devices - Your model is on the GPU but your data is on the CPU.
