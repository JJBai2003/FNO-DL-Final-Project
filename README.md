# FNO-DL-Final-Project
## Introduction
**Authors**: Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhat-tacharya, Andrew Stuart, Anima Anandkumar 
<br>
**Venue**: ICLR 2021
<br>
**Main Methods:**
The main method that we will re-implement is the Fourier Neural Operator, specifically applied to the 1D Burgers' Equation. The model is composed of the following layers: A linear layer to lift input to channel space, four consecutive layers of Fourier operators(1D convolution + Fourier convolution + skip connection + activation function), and another linear layer to project data back to the real dimension. 
<br>
**Contribution**: Partial differential equations (PDEs) are used to solve many real-life problems ranging from quantum dynamics to civil engineering. However, it is extremely difficult to identify the underlying complex functions of many variables for these problems. Machine learning offers a novel approach that allows scientists to learn functions without necessarily accounting for all of the hidden variables. However, there are still issues such as over-simplified models and intractable realistic simulations due to the overly complicated non-linear equations.
As a result, the authors of this paper propose a neural operator framework, which learns a mapping from one function space to another and one can evaluate the output function on the domain to get the classical mapping. FNO is special for its discretization invariance and universal approximation. It is the first model that solves the family of Navier-Stokes equations using a resolution-invariant approach, as previous models using Graph-based neural operators failed to converge. In addition, FNO can efficiently produce high-quality results with the same learned network parameters regardless of the discretization used on the input and output space, allowing it to perform zero-shot on high-resolution data after training on low-resolution data.
## Chosen Result
We aim to reproduce the following table, which contains the l2 error of the model's performance on 1D Burgers' Equation. This table demonstrates the power of FNO in solving partial differential equations in comparison to other State-of-the-Art models. We can see that the l2 error is significantly lower for FNO, and more importantly, the error doesn't vary much across different resolution(s). 
<img width="427" alt="l2_paper_result" src="https://github.com/JJBai2003/FNO-DL-Final-Project/assets/60070699/de44468d-6def-4836-9920-b0aa4b32e630">

## Re-implementation Details

### Data Generation
We were fortunately able to use the dataset provided by the researchers directly with the approval from instructors. Theoretically, the data are generated by solving the 1d Burgers' Equation using the pseudo-spectral split-step method with s = 8192. The link to accessing the data can be found here:https://ssd.mathworks.com/supportfiles/nnet/data/burgers1d/burgers_data_R10.mat. To run the training and testing jupyter notebook file correctly, please put the data file inside the same directory.

### FNO Block
Step 1: Lifting function to channel space
We used the torch.Linear method to lift the input from input space to channel space
<br>
Step 2: FNO Block
The FNO Block is composed of the following:
1. 1D convolution: We used the torch _Conv1D_ method 
2. Fourier convolution: This is the main component of the FNO Block
<tab> - The Fourier convolution layer takes the input and performs the Fourier transform using the integral operator. 
<tab> - This is done using the torch _einsum_ method
3. Skip connection: We generate the output of this layer by adding the output of the previous 2 layers. 
4. Activation function: we used ReLU as our non-linear activation function.
<br>
Step 3: Projecting output back to the target space
We used another Linear layer to project the output back to the target space.

### Training
We implemented training following the template from previous class projects with hyperparameters stated in the paper. Specifically, our dataset contains 1000 training data and 200 test data with batch size 16. We used the Adam optimizer with a learning rate of 0.001, a weight decay of 1e-4 with a scheduler of step size 50, and gamma equals 0.5.
The paper set epochs to be 500, but we don't have the same hardware resource as them so we ran for 50 epochs. 

### Testing
We implemented testing following the template from previous class projects. We loaded the test dataset and computed the average L2 loss. 

## Results and Analysis
Here is the table comparing the result of our implementation to the paper. 
![image](https://github.com/JJBai2003/FNO-DL-Final-Project/assets/60070699/7b0f6dbd-e1c8-4388-92d1-7eaa352351db)
- Our L2 losses were slightly higher than the result in the paper, which makes sense as we only trained for 50 epochs instead of 500.
- Despite the slight difference, our losses are relatively consistent throughout the resolutions, which means our implementation also achieved resolution-invariance.
- The error rate is very low, which means our implementation was able to learn the Burgers' equation and make accurate predictions at time = 1 for different resolutions.
- The training process was relatively fast, even for s = 8192, which reiterates the power and efficiency of FNOs.
- 
To better visualize the result, we created this graph, which displays a very similar trend to the paper.
![alt text](results\graph.png)
We generated the graph on the left by calling our model on a set of data(x) and then plotting the output. The graph on the right is the ground truth y. 


## Conclusion
We thought the paper did a great job laying out the architecture of the Fourier Neural Operator. We were able to extract the necessary information smoothly. However, we had a big misunderstanding of the paper initially which caused us to overcomplicate our implementations and significantly slowed us down. We originally thought we would be implementing a model to predict the result of the Burgers' Equation at every time stamp, but the mapping we are learning is initial data and evolution via Burgers' equation to the resulting function on [0,1] at time = 1. We learned that a mathematical understanding is very important for implementation, but sometimes theoretical understanding is only gained when you try to implement that understanding. After discovering our misunderstanding, we adjusted our game plan and were able to produce significant results. 
Some future directions that we can look into are using FNO to learn evolution via Burgers' equation at different times, and potentially other equations mentioned in the paper. Recently, another paper called the "Discretization Error of Fourier Neural Operators" was published(https://arxiv.org/abs/2405.02221). As a future step, it would be cool to look into this paper as well!

## References
- Zongyi Li, Nikola B. Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew M. Stuart, and Anima Anandkuma. Fourier Neural Operator for Parametric Partial Differential Equations. URL: https://arxiv.org/pdf/2010.08895 
- Project website: https://zongyi-li.github.io/blog/2020/fourier-pde/
- Dataset: https://ssd.mathworks.com/supportfiles/nnet/data/burgers1d/burgers_data_R10.mat
