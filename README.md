[Website] :- https://digit-predictor-jay.onrender.com/

It will take 20s to 40s to load the website some tiem it will take 6 min if the render where i deploy the project suspended the website due to not active so it willdepoly it again it will take almost 6 min


Here is use CNN model with MINST dataset the arch :-

Conv2d(input,32, kernel_size=3),
Relu(),
nn.ReLU()
MaxPool2d(kernel_size=2, stride=2),
Conv2d(32,64, kernel_size=3),
ReLU(),
MaxPool2d(kernel_size=2, stride=2),


Flatten(),
Linear(64*5*5, 128),
ReLU(),
Linear(128, 10),
