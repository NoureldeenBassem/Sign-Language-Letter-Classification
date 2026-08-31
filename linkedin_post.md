# LinkedIn post

I taught a model to read the alphabet off a hand. 🤟

Real time captioning for American Sign Language has to start somewhere, and that somewhere is one still photo of one hand shape mapped to one letter. Get it right and a fingerspelling reader can sit on top of it. Get it wrong and everything above inherits the error.

So I built that piece: a CNN designed and trained from scratch, no pretrained backbone, on the Sign Language MNIST dataset. 24 letters, the alphabet without J and Z since both need movement and cannot be caught in a still frame. 99.58% on 7,172 test images it never saw during training or tuning, plus a Streamlit app where you upload a photo and watch it predict.

The interesting part: 30 wrong answers out of 7,172, and 22 of them are the same mistake. Y read as L. Both signs are a fist with two things sticking out at roughly a right angle, thumb and little finger for Y, thumb and index for L. At 28x28 pixels the only thing separating them is which arm is longer, and a slight rotation erases even that. The other 8 errors are M read as N. Every other letter scored a perfect 1.000.

Before training I tried to guess where it would fail, ranking letter pairs by how similar their average images were. R, U, V and W came out on top. The model handled them perfectly. Similarity told me where to look, not what would break.

Designing the architecture instead of borrowing one meant every choice needed a reason. 28x28 is small, so I could afford only two downsamplings. I kept the 7x7 feature map and flattened it rather than using global average pooling, because for letters that are all closed fists, where the thumb sits IS the signal, and pooling throws out the answer.

The counterintuitive one: less augmentation, not more. This dataset was itself built by augmenting a few hundred original photographs, so pushing hard on rotation and zoom sends images somewhere the test set never goes. And horizontal flips are simply wrong here, mirroring a hand sign changes what it means.

Which is also the honest caveat. Train and test come from the same small pool of source photos, so 99.58% measures the dataset as much as the model. Point a phone at a real hand and it gets harder fast. So the app shows the 28x28 image the model actually sees next to the top three letters, rather than one confident looking answer.

Chasing the last 0.4% mattered less than knowing exactly which two letters it lives in.

Code 👉 https://github.com/NoureldeenBassem/Sign-Language-Letter-Classification

Try it 👉 https://sign-language-letter-classification-obclwkjrktzdkgwqmfqzuv.streamlit.app/

Big thanks to Waled Saied, my instructor, and Iyad Mahdy, my mentor, for guidance that shaped how I approached this.

Feedback is welcome, especially from anyone who has worked with classes this visually close.

#MachineLearning #DeepLearning #ComputerVision #CNN
