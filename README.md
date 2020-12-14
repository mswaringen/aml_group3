## Applied Machine Learning - Group 3 - Not_Hot_Dog_v2

Summary: Transfer learning via fine-tuning on EfficientNet

-   Data augmentation:
	- random rotation, random translation, random flip, random contrast
    

-   Pass 1:
	- Model created from EfficientNet
	- Remove top layer, freeze all weights, then recreate top layer
    

-   Pass 2:
	- Unfreeze top 20 layers excluding BatchNormalization
    
-   Results
		- Model Validation Acc: 0.6887
		- Kaggle Test-set Acc: 0.6890
    

  
  

Modifications:

-   Starting with Noisy Student weights
    
-   Increased drop out to 0.5
    
-   Learning rate uses warm-up period followed by cosine decay
    
-   Utilize EarlyStopping to train model in each pass to its optimum (val_loss stagnation)
    
-   Used progressively larger versions of EfficientNet, starting at 0 and moving 5, with requisite up-sampling of image resolution to 456x456
    

  
  

Explanation of Model Performance

-   Initial attempts, models included ResNet 50, Xception, EfficientNet
    

-   Transfer learning via Feature Extraction
    
-   Transfer learning via Fine Tuning
    
-   Train from scratch
    

-   Initial efforts produced validation acc in low 40s, overfitting a big issue as training acc in the 60-70s
    
-   Additional data augmentation helped some, but biggest gains from increasing model drop out to 0.5 from 0.2 and from moving to Noisy Student initial weights instead of ImageNet-> valid acc improved to low 60s
    

-   Lots of “bad” images require an increase in noise (augmentation and drop out) to better generalize, Noisy Student weights were built for this.
    
-   Validation acc now exceeded test acc by a wide margin
    

-   Final modifications of learning rate warm-up and decay were marginally useful, but saw bigger gains from upsampling image size and using a large EfficientNet model. The final model used EfficientNetB5 saw valid acc improve to 68.9%
    

  
  

Future Research

-   Grid Search Hyperparameters (already found nice values manually, didn’t both with this)
    

-   [https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)
    
-   [https://github.com/keras-team/keras-tuner](https://github.com/keras-team/keras-tuner)
    

-   Implement Noisy Student model (lots of work + need to include/create unlabeled data) [https://memotut.com/keras-implement-noisy-student-and-check-the-effect-a1559/](https://memotut.com/keras-implement-noisy-student-and-check-the-effect-a1559/)
    
-   Use Mixup Training (very promising but couldn’t make it work effectively, very slow training probably due to how I was loading the data) [https://www.dlology.com/blog/how-to-do-mixup-training-from-image-files-in-keras/](https://www.dlology.com/blog/how-to-do-mixup-training-from-image-files-in-keras/)
    
-   Use VisionTransformer (very promising but Keras implementation kept breaking on experimental packages, official Jax implementation required learning another framework)
    

-   [https://paperswithcode.com/paper/an-image-is-worth-16x16-words-transformers-1](https://paperswithcode.com/paper/an-image-is-worth-16x16-words-transformers-1)
    
-   [https://github.com/tuvovan/Vision_Transformer_Keras](https://github.com/tuvovan/Vision_Transformer_Keras)
    

-   Use a GAN (very cool but lots of data engineering, need to generate images, store them, reload by batch, etc)
    

-   [https://machinelearningmastery.com/semi-supervised-generative-adversarial-network/](https://machinelearningmastery.com/semi-supervised-generative-adversarial-network/)
    

  
  

Sources/More Info:

-   Fine-tuning with Keras
    

-   [https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/)
    

-   EfficientNet - Compound Model Scaling: A Better Way to Scale Up CNNs
    

-   [https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html)
    
-   [https://medium.com/@nainaakash012/efficientnet-rethinking-model-scaling-for-convolutional-neural-networks-92941c5bfb95](https://medium.com/@nainaakash012/efficientnet-rethinking-model-scaling-for-convolutional-neural-networks-92941c5bfb95)
    

-   Bag of Tricks for Image Classification
    

-   [https://www.dlology.com/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/](https://www.dlology.com/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/)
    
-   [https://paperswithcode.com/paper/bag-of-tricks-for-image-classification-with](https://paperswithcode.com/paper/bag-of-tricks-for-image-classification-with)
    

-   Loading Data in Keras from DataFrame [https://vijayabhaskar96.medium.com/tutorial-on-keras-flow-from-dataframe-1fd4493d237c](https://vijayabhaskar96.medium.com/tutorial-on-keras-flow-from-dataframe-1fd4493d237c)
    
-   Noisy Student Explained [https://paperswithcode.com/method/noisy-student](https://paperswithcode.com/method/noisy-student)
