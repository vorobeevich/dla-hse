# Report

Firstly, I did not implement model streaming due to compatibility issues with the ffmpeg library. So, I only worked on the compression of the model.
Also, in the main.ipynb notebook, I left only the most successful final experiments. I describe about my unsuccessful attempts in the report.

My experiments:

1. **Dark knowledge distillation**

As a first experiment, I decided to do a distillation based only on the outputs of the model. To do this, 
I used KL divergence between student and teacher model probabilities, added to the usual student cross entropy.
After some time iterating over the parameters, I realized that it is not necessary to greatly reduce the hidden size. 
You can reduce the number of filters in the convolution, leave only 1 gru layer, increase the stride, and reduce the hidden size just a little. 
Then we get a well-compressed model that still beats the desired quality.
At first, I tried to train the model on 20 epochs, but for a small model, this is too small. 
I tried to overcome the fact that the quality behaves more noisily than in the case of a large model by 
using a scheduler and softmax temperature, but was not successful. 
Therefore, I first increased the number of epochs to 100, and then left it at 50.

In the next picture, we see figures of quality and losses depending on the number of epochs.
![image](https://user-images.githubusercontent.com/61282340/200200428-3db2e0aa-1d89-42cf-8dc6-af2fc41d386b.png)
The behavior of all three charts looks similar: the metric and losses are a little noisy, but gradually decrease.

2. **Fp32 -> Fp16**

For this experiment, I only tested the model on 20 epochs to quickly check its performance.

![image](https://user-images.githubusercontent.com/61282340/200200514-1261562c-6daf-4623-9017-f7fa61a66379.png)

The graph of the metric is very noisy: the compressed model simply cannot train well. Therefore, I immediately moved on to the next experiment.

3. **Final setup**

In the end, I combined both approaches, and it gave me unexpectedly good quality - I achieved the desired result in just 20 epochs.
I wasn't able to run other experiments like int8 quantization, pruning, or attenuation distillation due to various bugs that I didn't have time to fix.

4. **My score**

And so, I used distillation on a model with a compressed number of parameters and data type fp16.
![image](https://user-images.githubusercontent.com/61282340/200200617-a3cd192f-6b25-4a2b-9194-9c262e91be41.png)

![image](https://user-images.githubusercontent.com/61282340/200200658-a5fe880b-120e-48bd-8793-85110817444c.png)

**(7.4 + 6.4) × 7 / 20 = 4.83**
