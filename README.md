# GAN_Transformer
---

### The project at hand takes a baseline Generative Adversarial Network, as well as an altered version of this network that contains a transformer within the discriminator portion of the GAN, performed on image datasets. 

#### Here are the two python files you need: [gan.py](gan.py), [transGAN.py](transGAN.py)
#### With these two files you can train the two architectures on the Cifar10 dataset, and the resulting weights will be saved.
---

### The desired result is to show the effect that the transformer adds to the results, achieving better results with shorter computational time.
#### The next two important files will be notebooks that recreate the architecture, load in the weights, and display the resulting generated images: [gan_generate.ipynb](gan_generate.ipynb), [transGAN_generate.ipynb](transGAN_generate.ipynb)
---

### The .sh files are shell files used as wrappers so that the .py files can be ran through singularity using the container image provided by my professor. 
---

### If you want to run these tests with your own image files I will leave some documentation and files here to address the changes necessary to do so: [customGAN](custom_dataset_gan.py), [customtransGAN](custom_dataset_transGAN.py), [generateGAN](custom_dataset_train_gan.ipynb), [generatetransGAN](custom_dataset_train_transGAN.ipynb)
