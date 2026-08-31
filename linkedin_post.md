# LinkedIn post

I taught a model to read the alphabet off a hand. 🤟

The problem: real time captioning for American Sign Language has to start somewhere, and the somewhere is one still photo of one hand shape mapped to one letter. Get that right and a fingerspelling reader can sit on top of it, taking a sequence of shapes and turning it into text. Get it wrong and everything above it inherits the error.

What I built: a CNN designed and trained from scratch, no pretrained backbone, on the Sign Language MNIST dataset. 27,455 training images, 24 letters, the alphabet without J and Z since both need movement and cannot be caught in a still frame. It gets 99.58% right on 7,172 test images it never saw during training or tuning, and I put it in a Streamlit app where you upload a photo and watch it predict.

The part I found interesting: 30 wrong answers out of 7,172, and 22 of them are the same mistake. Y read as L. Both signs are a fist with two things sticking out at roughly a right angle, thumb and little finger for Y, thumb and index for L. At 28x28 pixels the only thing separating them is which arm of that L shape is longer, and a slight rotation erases even that. The other 8 errors are M read as N, two fists that differ by whether the thumb sits under two fingers or three. Every other letter scored a perfect 1.000.

Before training I tried to guess where it would fail. I averaged every image of each class and ranked the pairs by similarity: R, U, V and W came out on top, all of them two straight fingers held up. The model handled them perfectly. So visual similarity told me where to look, not what would break, which is a useful thing to be wrong about.

Designing the architecture instead of borrowing one meant every choice needed a reason. 28x28 is small, so I could only afford two downsamplings before the feature map stopped meaning anything. I kept the 7x7 spatial map and flattened it rather than using global average pooling, because for a family of letters that are all closed fists, *where* the thumb is IS the signal, and pooling it away throws out the answer. Batch norm after every convolution is what made a randomly initialized network trainable at a 1e-3 learning rate.

The counterintuitive one: I used less augmentation, not more. This dataset was itself built by augmenting a few hundred original photographs, so pushing hard on rotation and zoom sends images somewhere the test set never goes. And horizontal flips are simply wrong here, mirroring a hand sign changes what it means.

Which is also the honest caveat. Train and test come from the same small pool of source photographs, so 99.58% measures the dataset as much as the model. Point a phone at a real hand, with a busy background and different lighting, and it gets harder fast. So the app shows the 28x28 image the model actually sees and the top three letters with their probabilities, rather than one confident looking answer. Seeing what the model sees explains most of its mistakes.

Under the hood: TensorFlow/Keras, three convolutional stages of 3x3 filters (32 to 64 to 128) with batch norm, dropout growing 0.25 to 0.5, Adam with early stopping and a learning rate schedule, evaluated with macro F1 and balanced accuracy on a held out test set rather than accuracy alone.

Chasing the last 0.4% mattered less than knowing exactly which two letters it lives in.

Code 👉 https://github.com/NoureldeenBassem/Sign-Language-Letter-Classification

Try it 👉 https://sign-language-letter-classification-obclwkjrktzdkgwqmfqzuv.streamlit.app/

Big thanks to [INSTRUCTOR NAME], my instructor, and [MENTOR NAME], my mentor, for guidance that shaped how I approached this.

Feedback is welcome, especially from anyone who has worked with sign language data or classes this visually close.

#MachineLearning #DeepLearning #ComputerVision #CNN #AccessibleTech
