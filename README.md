# CNN-CPP Framework

## About the CNN-CPP

### Introduction

CNN-CPP is a comprehensive framework written from scratch in C++ for building and training Convolutional Neural Networks
(**CNNs**). Designed for flexibility and modularity, it provides an environment for creating deep learning models, with
a focus on image processing and classification. Utilizing modern C++ features and libraries such as Eigen for Tensor
operations and OpenCV for image handling, CNN-CPP aims to deliver a powerful toolset for deep learning applications.

### Inspiration

Inspired by TensorFlow, CNN-CPP seeks to replicate its versatility and power within a C++ framework. It provides C++
users with similar capabilities to define, train, and deploy neural network models effectively.

### Goals

- **Efficiency**: Maximize performance and minimize resource usage through optimized C++ implementations.
- **Flexibility**: Allow users to easily define, train, and modify complex neural network architectures.
- **Comprehensiveness**: Provide a complete suite of tools and functionalities for handling the entire deep learning
  workflow, from data loading and augmentation to model evaluation and learning visualization.
- **Extensibility**: Enable developers to extend the framework with custom layers, optimizers, and other components,
  facilitating research and experimentation.
- **Cross-Platform Compatibility**: Support multiple operating systems, including Linux, macOS, and Windows, through
  dockerization.

## Repository Structure

```plaintext
CNN-CPP/
├── datasets/                   # Versioned Datasets
├── external/                   # External dependencies for the project
│   ├── eigen/                  # Eigen library for tensor operations
│   ├── googletest/             # GoogleTest for unit testing
│   └── opencv/                 # OpenCV for image processing
├── include/                    # Header files for the project
│   ├── CNN/                    # Definitions for Convolutional Neural Network
│   ├── Common/                 # Common utilities and helper functions
│   ├── Image/                  # Definitions for image processing
│   └── ThreadPool/             # Thread pool for parallel execution
├── logs/                       # Placeholder directory for log files
├── plots/                      # Placeholder directory for storing plots and graphs
├── src/                        # Source files for the project
│   ├── CNN/                    # Source files for Convolutional Neural Network
│   ├── Common/                 # Source files for common utilities
│   ├── Image/                  # Source files for image processing
│   ├── ThreadPool/             # Source files for thread pool implementation
│   └── main.cpp                # Main program file
├── tests/                      # Unit tests for the project
├── wmodels/                    # Already created models, used in Results
├── .gitignore                  # Specifies untracked files to ignore
├── CMakeLists.txt              # CMake configuration file
├── Dockerfile                  # Docker configuration for setting up the environment
├── install.sh                  # Script for installing the Linux dependencies
├── README.md                   # Project overview and setup instructions
├── tasks.py                    # Script for automating common tasks
└── tools.py                    # Utilities and helper tools for tasks
```

## Framework Components & Capabilities

### Batch Manager

The BatchManager class serves as the input layer to the neural network by managing the batching of images and labels. It
facilitates efficient data feeding during training and testing by handling the following tasks:

- <code style="color: teal;">Batch Initialization:</code> The BatchManager initializes batches based on the input data,
  categorizing images and labels for training or testing.

- <code style="color: teal;">Data Shuffling:</code> It implements dataset shuffling to ensure that the batches are
  randomized for each epoch, which helps in achieving better generalization in the model.

- <code style="color: teal;">Batch Retrieval:</code> The class provides functionality to retrieve the next batch of
  images and labels for processing. This includes one-hot encoding of labels and maintaining batch balance across
  categories.

- <code style="color: teal;">Single Prediction Support:</code> It supports loading batches specifically for single
  prediction tasks, allowing for inference on individual images.

- <code style="color: teal;">Batch Indexing and Balancing:</code> The manager keeps track of the current batch index and
  ensures that batches are balanced in terms of category distribution. If last batch is not full, it fills the remaining
  slots with random images from the original dataset to maintain batch size consistency.

- <code style="color: teal;">Batch Saving:</code> Optionally, batches can be saved to disk for debugging or analysis,
  with each batch organized by category.

- <code style="color: teal;">Support for Different Batch Types:</code> The BatchManager can handle different batch
  types, including training and testing batches, by selectively retrieving images and labels according to the batch type
  specified.

- <code style="color: teal;">Category Management:</code> It manages category information, allowing retrieval of category
  names and ensuring that all operations are consistent with the dataset's categorical structure.

---

### Convolution Layer

The ConvolutionLayer class implements a convolutional layer for neural networks, allowing the network to extract spatial
features from input images. This layer supports several key functionalities:

- <code style="color: teal;">Initialization: </code>

  - <code style="color: SteelBlue;">Number of filters</code>
  - <code style="color: SteelBlue;">Kernel size</code>
  - <code style="color: SteelBlue;">Stride</code>
  - <code style="color: SteelBlue;">Padding</code>
  - <code style="color: SteelBlue;">Kernel initialization:</code>

    - <code style="color: SkyBlue;">He</code>
      <h2 style="display: inline-block;">$$\text{Kernels} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{\text{input depth} \times \text{kernel size}^2}}\right)$$</h2>
    - <code style="color: SkyBlue;">Xavier</code>
      <h2 style="display: inline-block;">$$\text{Kernels} \sim \mathcal{N}\left(0, \sqrt{\frac{1}{\text{input depth} \times \text{kernel size}^2}}\right)$$</h2>
    - <code style="color: SkyBlue;">Random Normal</code>
      <h2 style="display: inline-block;">$$\text{Kernels} \sim \mathcal{N}(0, 1)$$</h2>

  - <code style="color: SteelBlue;">Bias initialization:</code>
    - <code style="color: SkyBlue;">Zero</code>
      <h2 style="display: inline-block;">$$\text{Bias} \ 0$$</h2>
    - <code style="color: SkyBlue;">Random Normal</code>
      <h2 style="display: inline-block;">$$\text{Bias} \ \mathcal{N}(0, 1)$$</h2>

- <code style="color: teal;">Forward Pass</code> The layer performs convolution operations on the input batch, computing
  feature maps using the specified kernels and biases. It uses multi-threading to parallelize the forward pass for
  improved performance.

1. **Convolution**

   <h3 style="display: inline-block;">$$\text{Feature Map}(f, i, j) = \sum_{d=0}^{D-1} \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} \text{Input}(d, i \cdot \text{stride} + m, j \cdot \text{stride} + n) \cdot \text{Kernel}(f, d, m, n)$$</h3>

   - $\text{Feature Map}(f, i, j)$ is the result of the convolution for filter $f$ at position $(i, j)$.
   - $D$ is the input depth (number of input channels).
   - $K$ is the kernel size.
   - $\text{Input}(d, i \cdot \text{stride} + m, j \cdot \text{stride} + n)$ represents the input patch.
   - $\text{Kernel}(f, d, m, n)$ is the kernel weight for filter $f$, depth $d$, and position $(m, n)$.

2. **Adding Biases**

   <h3 style="display: inline-block;">$$\text{Output}(f, i, j) = \text{Feature Map}(f, i, j) + \text{Bias}(f)$$</h3>

   - $\text{Output}(f, i, j)$ is the final output value after adding bias.
   - $\text{Bias}(f)$ is the bias term added to the feature map for filter $f$.

- <code style="color: teal;">Backward Pass</code> The backward method calculates gradients with respect to the kernels,
  biases, and input data, which are used for updating the layer's parameters. It supports parallel processing to
  efficiently compute gradients for the entire batch.

1. **Gradient with Respect to Input:**

   <h3 style="display: inline-block;">$$\text{dInput}(d, i, j) = \sum_{f=0}^{F-1} \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} \text{dOutput}(f, i - m, j - n) \cdot \text{Kernel}(f, d, m, n)$$</h3>

   - $\text{dInput}(d, i, j)$ is the gradient of the loss with respect to the input at depth $d$, position $(i, j)$.
   - $F$ is the number of filters.
   - $\text{dOutput}(f, i - m, j - n)$ is the gradient of the loss with respect to the output.
   - $\text{Kernel}(f, d, m, n)$ is the kernel weight for filter $f$, depth $d$, and position $(m, n)$.

2. **Gradient with Respect to Kernels:**

   <h3 style="display: inline-block;">$$\text{dKernel}(f, d, m, n) = \sum_{i=0}^{H-1} \sum_{j=0}^{W-1} \text{dOutput}(f, i, j) \cdot \text{Input}(d, i \cdot \text{stride} + m, j \cdot \text{stride} + n)$$</h3>

   - $\text{dKernel}(f, d, m, n)$ is the gradient of the loss with respect to the kernel at filter $f$, depth $d$, and
     position $(m, n)$.
   - $H$ and $W$ are the height and width of the output feature map.
   - $\text{dOutput}(f, i, j)$ is the gradient of the loss with respect to the output.

3. **Gradient with Respect to Bias:**

   <h3 style="display: inline-block;">$$\text{dBias}(f) = \sum_{i=0}^{H-1} \sum_{j=0}^{W-1} \text{dOutput}(f, i, j)$$</h3>

   - $\text{dBias}(f)$ is the gradient of the loss with respect to the bias for filter $f$.
   - $\text{dOutput}(f, i, j)$ is the gradient of the loss with respect to the output.

- <code style="color: teal;">Padding and Convolution</code> The layer handles padding of input images, ensuring correct
  spatial dimensions for output feature maps. The convolve method performs the element-wise multiplication of the input
  and kernel tensors over the receptive field, returning the convolution result.

1. **Padding:**

   <h3 style="display: inline-block;">$$\text{padded-Input}(i + \text{pad}, j + \text{pad}) = \text{input}(i, j)$$</h3>

   - $\text{padded-Input}:$ This is the output tensor after padding is applied.
   - $\text{pad}$: The amount of padding added to each side of the input tensor.
   - $\text{input}$: The original input tensor before padding.
   - $i, j$: Indices of the original input tensor.

2. **Convolve:**

   <h3 style="display: inline-block;">$$\text{Sum} = \sum_{i=0}^{\text{kernel\_height} - 1} \sum_{j=0}^{\text{kernel\_width} - 1} \text{input}(\text{start\_row} + i, \text{start\_col} + j) \times \text{kernel}(i, j)$$</h3>

   - $\text{Sum}$: This is the result of the convolution operation.
   - $\text{Kernel Height}$: The height of the kernel matrix.
   - $\text{Kernel Width}$: The width of the kernel matrix.
   - $\text{Input}$: The input tensor being convolved.
   - $\text{Kernel}$: The kernel or filter applied to the input.
   - $\text{Start Row/Column}$: The starting positions in the input tensor from where the kernel is applied.

- <code style="color: teal;">Output Height and Width Calculation</code>:

  <h3 style="display: inline-block;">$$\text{Output H/W} = \left\lfloor \frac{\text{Input H/W} - \text{Kernel Size} + 2 \times \text{Padding}}{\text{Stride}} \right\rfloor + 1$$</h3>

- <code style="color: teal;">Thread Pool</code> It utilizes thread pools for both forward and backward passes,
  distributing computation across available CPU cores (hardware threads) to optimize processing speed.

- <code style="color: teal;">Parameter Management</code> The class provides methods for setting and retrieving kernels
  and biases, allowing external manipulation and inspection. It ensures that these parameters can be loaded during
  training (ELRALES).

- <code style="color: teal;">Error Handling</code> It provides error detection mechanism:

  - Throws exception if filters, kernel size, stride, or padding are invalid during initialization.
  - Validates output dimensions during the forward pass, ensuring they are positive.
  - Ensures padding values are non-negative and appropriately sized.
  - Logs errors when setting kernels and biases if dimensions mismatch.
  - Checks for dimension mismatches in setKernels and setBiases, throwing exceptions if necessary.

---

### Max Pooling Layer

The MaxPoolingLayer class implements a max pooling layer for neural networks, which reduces the spatial dimensions of
the input while retaining the most important features. This class is crucial for down-sampling input feature maps,
leading to a more efficient and less complex network.

- <code style="color: teal;">Initialization:</code>

  - <code style="color: SteelBlue;">Pool Size</code>
  - <code style="color: SteelBlue;">Stride</code>

- <code style="color: teal;">Forward Pass</code> The forward method applies the max pooling operation on each input
  batch, extracting the maximum value from each pooling window to form the output feature maps. It also stores the
  indices of these maximum values for use during backpropagation.

  <h3 style="display: inline-block;">$$\text{MaxPooling}(i, j) = \max_{m=0}^{\text{Pool Size}-1} \max_{n=0}^{\text{Pool Size}-1} \text{Input}(i \cdot \text{Stride} + m, j \cdot \text{Stride} + n)$$</h3>

  - $i, j$: Indices of the output feature map.
  - $m, n$: Iterators over the pooling window, each ranging from $0$ to $\text{Pool Size} - 1$.
  - $\text{Stride}$: The stride of the pooling operation.

- <code style="color: teal;">Backward Pass</code> During backpropagation, the backward method uses the stored indices to
  propagate gradients back to the input layer. It ensures that only the positions corresponding to the maximum values
  from the forward pass receive non-zero gradients, which helps in updating weights effectively.

  <h3 style="display: inline-block;">$$\text{Gradient}(i, j) = \text{dOutput}(i, j) \text{ if } (i, j) \text{ is the max index, otherwise } 0$$</h3>

  - $\text{dOutput}(i, j)$: The gradient of the loss with respect to the output at position $(i, j)$.
  - $\text{Gradients}$: Routed only to those indices that correspond to the maximum values used in the forward pass, and
    all other gradients are set to zero.

- <code style="color: teal;">Index Management</code> The class maintains indices of maximum values for each pooling
  window in each input image, facilitating accurate gradient propagation during the backward pass.

- <code style="color: teal;">Output Height and Width Calculation</code>

  <h3 style="display: inline-block;">$$\text{Output H/W} = \left\lfloor \frac{\text{Input H/W} - \text{Pool Size}}{\text{Stride}} \right\rfloor + 1$$</h3>

- <code style="color: teal;">Error Handling</code> It checks for invalid pooling configurations by ensuring that the
  calculated output dimensions are positive. If not, it throws an exception indicating potentially incompatible pool
  size or stride settings.

---

### Average Pooling Layer

The AveragePoolingLayer class implements an average pooling layer for neural networks, which reduces the spatial
dimensions of input feature maps by computing the average of elements within a defined pooling window. This process
helps to down-sample the input, reducing the number of parameters and computations in the network, while maintaining
essential spatial information.

- <code style="color: teal;">Initialization:</code>
- <code style="color: SteelBlue;">Pool Size</code>
- <code style="color: SteelBlue;">Stride</code>

- <code style="color: teal;">Forward Pass</code> The forward method applies the average pooling operation to each input
  batch, calculating the average of the values in each pooling window and forming the output feature maps. This method
  is essential for down-sampling and reducing the spatial dimensions of the input data while preserving important
  feature information.

  <h3 style="display: inline-block;">$$\text{AveragePooling}(i, j) = \frac{1}{\text{Pool Size}^2} \sum_{m=0}^{\text{Pool Size}-1} \sum_{n=0}^{\text{Pool Size}-1} \text{Input}(i \cdot \text{Stride} + m, j \cdot \text{Stride} + n)$$</h3>

  - $i, j$: Indices of the output feature map.
  - $m, n$: Iterators over the pooling window, each ranging from $0$ to $\text{Pool Size} - 1$.
  - $\text{Stride}$: The stride of the pooling operation.

- <code style="color: teal;">Backward Pass</code> During backpropagation, the backward method computes the gradients for
  the input data. The gradient from the output is evenly distributed across the positions within the corresponding
  pooling window, enabling proper weight updates during training.

  <h3 style="display: inline-block;">$$\text{Gradient}(i, j) = \frac{\text{dOutput}(i, j)}{\text{Pool Size}^2}$$</h3>

  - $\text{dOutput}(i,j)$: The gradient of the loss with respect to the output at position $(i, j)$.
  - $\text{Pool Size}$: Pool Size is the total number of elements in the pooling window.

- <code style="color: teal;">Output Height and Width Calculation</code>
  <h3 style="display: inline-block;">$$\text{Output H/W} = \left\lfloor \frac{\text{Input H/W} - \text{Pool Size}}{\text{Stride}} \right\rfloor + 1$$</h3>

- <code style="color: teal;">Error Handling</code> The layer checks for valid configurations by ensuring that the
  calculated output dimensions are positive. It will throw an exception if invalid pooling size or stride values result
  in negative output dimensions.

---

### Activation Layer

The ActivationLayer class implements various activation functions used in neural networks to introduce non-linearity
into the model. This class offers flexibility by supporting multiple activation types and their respective derivatives,
which are essential for both the forward and backward passes of the network. The key features of this class are:

- <code style="color: teal;">Supported Activation Types:</code>

  - <code style="color: SteelBlue;">ReLU</code>

  <h3 style="display: inline-block;">$$\text{ReLU}(x) = \max(0, x)$$</h3>

  <h3 style="display: inline-block;">$$\text{ReLU}'(x) = \text{if } x > 0 \text{ then } 1 \text{ otherwise } 0$$</h3>

- <code style="color: SteelBlue;">Leaky ReLU</code>

  <h3 style="display: inline-block;">$$\text{Leaky ReLU}(x) = \text{if } x \geq 0 \text{ then } x \text{ else } \alpha x$$</h3>

  <h3 style="display: inline-block;">$$\text{Leaky ReLU}'(x) = \text{if } x \geq 0 \text{ then } 1 \text{ else } \alpha$$</h3>

- <code style="color: SteelBlue;">ELU</code>

  <h3 style="display: inline-block;">$$\text{ELU}(x) = \text{if } x \geq 0 \text{ then } x \text{ else } \alpha (\exp(x) - 1)$$</h3>

  <h3 style="display: inline-block;">$$\text{ELU}'(x) = \text{if } x \geq 0 \text{ then } 1 \text{ else } \alpha e^x$$</h3>

- <code style="color: SteelBlue;">Sigmoid</code>

  <h3 style="display: inline-block;">$$\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}$$</h3>

  <h3 style="display: inline-block;">$$\text{Sigmoid}'(x) = \text{Sigmoid}(x) \cdot (1 - \text{Sigmoid}(x))$$</h3>

- <code style="color: SteelBlue;">Softmax</code>

  <h3 style="display: inline-block;">$$\text{Softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^{N} \exp(x_j)}$$</h3>

  <h3 style="display: inline-block;">$$\text{Loss} = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)$$</h3>

  <h3 style="display: inline-block;">$$\frac{\partial \text{Loss}}{\partial x_i} = \hat{y}_i - y_i$$</h3>

- <code style="color: SteelBlue;">TanH</code>
  <h3 style="display: inline-block;">$$\text{Tanh}(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}$$</h3>

  <h3 style="display: inline-block;">$$\text{Tanh}'(x) = 1 - \text{Tanh}^2(x)$$</h3>

- <code style="color: teal;">Forward Pass</code> The forward method applies the chosen activation function to each
  element of the input tensor, allowing for both 2D and 4D input structures by wrapping and unwrapping tensors as
  needed.

- <code style="color: teal;">Backward Pass</code> Computes the derivative of the activation function for the backward
  propagation step, which is crucial for updating network parameters during training.

- <code style="color: teal;">Error Handling</code> The class checks for unsupported activation types and throws
  exceptions if such a type is encountered.

---

### Flatten Layer

The FlattenLayer class is designed to transform multi-dimensional input tensors into flat vectors, making it essential
for connecting convolutional layers to fully connected layers in a neural network. Here are its key functionalities:

- <code style="color: teal;">Forward Pass</code> The FlattenLayer reshapes the input tensor from (batch\*size, depth,
  height, width) to a 2D tensor with shape (batch\*size, flattened\*size), where flattened_size is the product of depth,
  height, and width. The output is returned as a 4D tensor with shape (batch_size, 1, 1, flattened_size).

  <h3 style="display: inline-block;">$$X\_{\text{out}} = \text{reshape}(X, (N, 1, 1, C \times H \times W))$$</h3>

  - $X_{\text{out}}$: The output tensor after flattening.
  - $X$: The input tensor with shape $(N, C, H, W)$ where $N$ is the batch size, $C$ is the number of channels, $H$ is
    the height, and $W$ is the width.
  - $\text{reshape}$: The operation used to flatten the input tensor $X$ into the desired output shape.

  - This formula represents the flattening operation performed in the `FlattenLayer` of the C++ code. The input tensor
    is reshaped from a 4D tensor to a flattened 2D tensor and then reshaped back to a 4D tensor with the flattened size
    in the last dimension.

- <code style="color: teal;">Backward Pass</code> It reshapes the gradient from the output tensor back to the original
  shape of the input tensor, allowing gradients to propagate correctly through the network.

  <h3 style="display: inline-block;">$$\text{dInput} = \text{reshape}(\text{dOutput}, (N, C, H, W))$$</h3>

  - $\text{dInput}$: The gradient with respect to the input, reshaped back to its original dimensions.
  - $\text{dOutput}$: The gradient of the loss with respect to the output of the flatten layer, typically a 4D tensor
    with shape $(N, 1, 1, C \times H \times W)$.
  - $\text{reshape}$: The operation used to convert the flattened gradient back to the original input shape.

This formula represents the backward pass in the `FlattenLayer`. The gradient $\text{dOutput}$ is reshaped back to match
the original dimensions of the input tensor $(N, C, H, W)$, effectively reversing the flattening operation performed
during the forward pass.

- <code style="color: teal;">Shape Management</code> The layer tracks the original dimensions of the input tensor to
  ensure accurate reshaping during both the forward and backward passes.

- <code style="color: teal;">Error Handling</code> The layer checks for consistency in reshaping operations and will
  throw an exception if there is a mismatch in dimensions during the backward pass.

---

### Fully Connected Layer

The FullyConnectedLayer class represents a dense layer in a neural network, where each neuron is connected to every
neuron in the previous layer. It is responsible for learning linear combinations of the input features, which are then
used for predictions or further transformations.

- <code style="color: teal;">Initialization:</code>

  - <code style="color: SteelBlue;">Weights:</code>

    - <code style="color: SkyBlue;">He</code>

      <h2 style="display: inline-block;">$$\text{Weights} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{\text{Input Size}}}\right)$$</h2>

    - <code style="color: SkyBlue;">Xavier</code>

      <h2 style="display: inline-block;">$$\text{Weights} \sim \mathcal{N}\left(0, \sqrt{\frac{1}{\text{Input Size}}}\right)$$</h2>

    - <code style="color: SkyBlue;">Random Normal</code>

      <h2 style="display: inline-block;">$$\text{Weights} \sim \mathcal{N}(0, 1)$$</h2>

  - <code style="color: SteelBlue;">Biases:</code>

    - <code style="color: SkyBlue;">Zero</code>

      <h2 style="display: inline-block;">$$\text{Bias} = 0$$</h2>

    - <code style="color: SkyBlue;">Random Normal</code>

      <h2 style="display: inline-block;">$$\text{Bias} \sim \mathcal{N}(0, 1)$$</h2>

- <code style="color: teal;">Optimizer Requirement</code> This layer requires an optimizer to update weights and biases
  during training, as it contains trainable parameters.

- <code style="color: teal;">Forward Pass</code> The forward method takes an input batch of 4D tensors, flattens it to a
  2D tensor, and performs matrix multiplication with the weights, followed by the addition of biases. The result is
  reshaped back into a 4D tensor with the shape (batch_size, 1, 1, output_size).

1. **Matrix Multiplication:**

   <h3 style="display: inline-block;">$$\text{output2d}(b, o) = \sum_{i=0}^{\text{inputSize}-1} \text{input2d}(b, i) \times \text{weights}(o, i)$$</h3>

   - $b$: is the batch index.
   - $o$: is the output neuron index.
   - $i$: is the input feature index.
   - $\text{input2d}(b, i)$: the value of the input feature $i$ for the batch $b$.
   - $\text{weights}(o, i)$: the weight associated with the connection between input feature $i$ and output neuron $o$.
   - $\text{output2d}(b, o)$: the resulting output for neuron $o$ in batch $b$.

1. **Adding Bias:**

   <h3 style="display: inline-block;">$$\text{output2d}(b, o) += \text{biases}(o)$$</h3>

   - $b$: is the batch index.
   - $o$: is the output neuron index.
   - $\text{output2d}(b, o)$: the computed output before adding the bias.
   - $\text{biases}(o)$: the bias for the output neuron $o$.

- <code style="color: teal;">Backward Pass</code> During the backward pass, the layer calculates the gradients of the
  loss with respect to its weights, biases, and inputs. These gradients are used to update the parameters using the
  optimizer.

1. **Gradient with respect to weights:**

   <h3 style="display: inline-block;">$$\text{dweights}(o, i) += \sum_{b=0}^{\text{batchSize}-1} \text{doutput2d}(b, o) \times \text{input2d}(b, i)$$</h3>

   - $b$: is the batch index.
   - $o$: is the output neuron index.
   - $i$: is the input feature index.
   - $\text{dweights}(o, i)$: is the gradient of the weight connecting input feature $i$ to output neuron $o$.
   - $\text{input2d}(b, i)$: the value of the input feature $i$ for the batch $b$.
   - $\text{doutput2d}(b, o)$: The partial derivative of the loss function with respect to the output of the neuron at
     index $o$ for the $b$-th example in the batch.

2. **Gradient with respect to biases:**

   <h3 style="display: inline-block;">$$\text{dbiases}(o) += \sum_{b=0}^{\text{batchSsize}-1} \text{doutput2d}(b, o)$$</h3>

   - $b$: is the batch index.
   - $o$: is the output neuron index.
   - $\text{input2d}(b, i)$: the value of the input feature $i$ for the batch $b$.
   - $\text{doutput2d}(b, o)$: The partial derivative of the loss function with respect to the output of the neuron at
     index $o$ for the $b$-th example in the batch.
   - $\text{dbiases}(o)$: is the gradient of the bias for output neuron $o$.

3. **Gradient with respect to inputs:**

   <h3 style="display: inline-block;">$$\text{dinput2d}(b, i) += \sum_{o=0}^{\text{outputSize}-1} \text{doutput2d}(b, o) \times \text{weights}(o, i)$$</h3>

   - $b$: is the batch index.
   - $o$: is the output neuron index.
   - $i$: is the input feature index.
   - $\text{dinput2d}(b, i)$: partial derivative of the loss function with respect to the input value at index $i$ for
     the $b$-th example in the batch. This gradient is propagated to the previous layer.
   - $\text{doutput2d}(b, o)$: The partial derivative of the loss function with respect to the output of the neuron at
     index $o$ for the $b$-th example in the batch.
   - $\text{weights}(o, i)$: weight connecting input feature $i$ to output neuron $o$.

- <code style="color: teal;">Parameter Management</code> TThe class provides methods for setting and retrieving weights
  and biases, allowing external manipulation and inspection. It ensures that these parameters can be loaded during
  training (ELRALES).

- <code style="color: teal;">Error Handling</code> The layer validates input and output dimensions during both forward
  and backward passes, throwing exceptions for any inconsistencies in the expected and actual dimensions. It ensures
  that both input and output sizes are positive integers, maintaining the integrity of the network configuration.

---

### Batch Normalization Layer

The BatchNormalizationLayer class implements batch normalization for neural networks, a technique used to stabilize and
accelerate training by normalizing the input of each layer. This layer helps improve the convergence rate and overall
performance of the network by maintaining mean and variance at stable levels. It sets up internal tensors for scaling
(gamma) and shifting (beta), as well as moving averages of mean and variance.

- <code style="color: teal;">Initialization:</code>

  - <code style="color: SkyBlue;">Epsilon</code>
  - <code style="color: SkyBlue;">Momentum</code>

- <code style="color: teal;">Forward Pass</code> The forward method computes the mean and variance for the current
  batch, normalizes the input data, and scales and shifts it using gamma and beta. It also updates the moving averages
  of mean and variance using the specified momentum.

  <h3 style="display: inline-block;">$$\mu = \frac{1}{N} \sum_{n=1}^{N} x_i^{(n)}$$</h3>

  <h3 style="display: inline-block;">$$\sigma^2 = \frac{1}{N} \sum_{n=1}^{N} (x_i^{(n)} - \mu)^2$$</h3>

  <h3 style="display: inline-block;">$$\hat{x}_i^{(n)} = \frac{x_i^{(n)} - \mu}{\sqrt{\sigma^2 + \epsilon}}$$</h3>

  <h3 style="display: inline-block;">$$y_i^{(n)} = \gamma \hat{x}_i^{(n)} + \beta$$</h3>

  - **$\mu$**: The mean ($\mu$) is computed as the average of the input values $x_i^{(n)}$ over the batch. This step
    centers the input data.

  - **$\sigma^2$**: The variance ($\sigma^2$) is calculated as the average of the squared differences between each input
    value and the mean. It measures how much the inputs vary from the mean.

  - **$\hat{x}_i^{(n)}$**: The normalized input ($\hat{x}_i^{(n)}$) is derived by subtracting the mean from each input
    and dividing by the square root of the variance plus a small constant $\epsilon$. This ensures the inputs have a
    mean of 0 and a variance of 1.

  - **$y_i^{(n)}$**: The final output ($y_i^{(n)}$) is obtained by scaling the normalized input with $\gamma$ and then
    shifting it with $\beta$. These parameters allow the model to adjust the normalized data to any desired scale and
    shift.

- <code style="color: teal;">Backward Pass</code> During backpropagation, the backward method computes gradients with
  respect to the input data, as well as the gamma and beta parameters. These gradients are used to update the parameters
  directly.

  <h3 style="display: inline-block;">$$\text{dgamma} = \sum_{n=1}^{N} \sum_{i=1}^{m} \text{doutput}(n, i) \times \hat{x}_i$$</h3>

  <h3 style="display: inline-block;">$$\text{dbeta} = \sum_{n=1}^{N} \sum_{i=1}^{m} \text{d\_output}(n, i)$$</h3>

  <h3 style="display: inline-block;">$$\text{dvariance} = \sum_{n=1}^{N} \sum_{i=1}^{m} \text{doutput}(n, i) \times (x_i - \mu) \times -\frac{1}{2} \times (\sigma^2 + \epsilon)^{-\frac{3}{2}}$$</h3>

  <h3 style="display: inline-block;">$$\text{dmean} = \sum_{n=1}^{N} \sum_{i=1}^{m} \left[\text{doutput}(n, i) \times -\frac{1}{\sqrt{\sigma^2 + \epsilon}}\right] + \text{dvariance} \times \frac{-2}{m} \times \sum_{i=1}^{m} (x_i - \mu)$$</h3>

  <h3 style="display: inline-block;">$$\text{dinput}(n, i) = \text{doutput}(n, i) \times \frac{1}{\sqrt{\sigma^2 + \epsilon}} + \frac{\text{dvariance} \times 2 \times (x_i - \mu)}{m} + \frac{\text{dmean}}{m}$$</h3>

  - **$\text{dgamma}$**: The gradient with respect to $\gamma$ (scale) is calculated by summing over the product of the
    gradient of the loss with respect to the output ($\text{doutput}$) and the normalized input ($\hat{x}_i$). This
    update reflects how the scaling factor $\gamma$ should change to minimize the loss.

  - **$\text{dbeta}$**: The gradient with respect to $\beta$ (shift) is simply the sum of the gradients of the loss with
    respect to the output ($\text{doutput}$). This indicates how the shift parameter $\beta$ should adjust.

  - **$\text{dvariance}$**: The gradient with respect to the variance $\sigma^2$ considers how changes in the variance
    affect the loss. This involves the sum of the product of the loss gradients, the difference between the input and
    the mean, and a factor dependent on the variance.

  - **$\text{dmean}$**: The gradient with respect to the mean $\mu$ is the sum of two components: the first part
    considers the direct impact of the mean on the loss, while the second part accounts for the influence of the
    variance on the mean.

  - **$\text{dinput}$**: The gradient with respect to the input $x_i$ is the sum of three terms: the first term scales
    the gradient by the inverse standard deviation, the second term accounts for the contribution of the variance, and
    the third term adjusts based on the mean.

- <code style="color: teal;">Parameter Management</code> The class provides methods for setting and retrieving gamma and
  beta, allowing external manipulation and inspection. It ensures that these parameters can be loaded during training
  (ELRALES).
- <code style="color: teal;">Error Handling</code> The layer includes checks to ensure that the initialization state is
  set correctly before performing forward or backward passes, preventing errors due to uninitialized parameters.

---

### Loss Function

The LossFunction class and its derived classes implement various loss functions used in neural networks to measure the
difference between predicted outputs and true targets. The key functionalities of each class are as follows:

- <code style="color: teal;">Binary Cross Entropy Loss</code> This loss function is used for binary classification
  tasks. It computes the loss using the binary cross-entropy formula and applies clipping to the predictions for
  numerical stability. The derivative method calculates the gradient of the loss with respect to the predictions.

  <h3 style="display: inline-block;">$$\text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \cdot \log(\hat{y}_i) + (1 - y_i) \cdot \log(1 - \hat{y}_i) \right]$$</h3>

  <h3 style="display: inline-block;">$$\text{doutput}(i) = \frac{\hat{y}_i - y_i}{\hat{y}_i \cdot (1 - \hat{y}_i)}$$</h3>

  - **$\text{Loss}$**: The Binary Cross Entropy loss function measures the difference between the predicted probability
    ($\hat{y}_i$) and the actual binary target ($y_i$) over a batch of size $N$. The loss is averaged over all examples
    in the batch.

  - **$\hat{y}_i$**: This represents the predicted probability for the $i$-th example in the batch, clipped to avoid
    taking the log of zero. It is given by the network's output.

  - **$y_i$**: This is the actual binary target (either 0 or 1) for the $i$-th example in the batch.

  - **$\text{doutput}(i)$**: The derivative of the loss function with respect to the network's output for the $i$-th
    example. It represents how much the loss would change with a small change in the network's output, guiding the
    backpropagation process to update the model's parameters accordingly.

- <code style="color: teal;">Mean Squared Error Loss</code> Used for regression tasks, this loss function computes the
  average squared difference between predictions and targets. The derivative method calculates the gradient of the loss
  with respect to the predictions.

  <h3 style="display: inline-block;">$$\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$</h3>

  <h3 style="display: inline-block;">$$\text{doutput}(i) = \frac{2 \cdot (\hat{y}_i - y_i)}{N}$$</h3>

  - **$\text{Loss}$**: The Mean Squared Error (MSE) loss function calculates the average squared difference between the
    predicted values ($\hat{y}_i$) and the actual target values ($y_i$) over a batch of size $N$.

  - **$\hat{y}_i$**: This represents the predicted value for the $i$-th example in the batch.

  - **$y_i$**: This is the actual target value for the $i$-th example in the batch.

  - **$\text{doutput}(i)$**: The derivative of the loss function with respect to the network's output for the $i$-th
    example. It indicates how much the loss would change with a small change in the network's output, guiding the
    backpropagation process to update the model's parameters.

- <code style="color: teal;">Categorical Cross Entropy Loss</code> This loss function is used for multi-class
  classification tasks with one-hot encoded targets. It calculates the loss using the categorical cross-entropy formula,
  summing the loss for each class where the target is 1. The derivative method calculates the gradient with respect to
  the predictions, assuming one-hot encoded targets.

  <h3 style="display: inline-block;">$$\text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{i,j} \cdot \log(\hat{y}_{i,j})$$</h3>

  <h3 style="display: inline-block;">$$\text{doutput}(i, j) = \hat{y}_{i,j} - y_{i,j}$$</h3>

  - **$\text{Loss}$**: The Categorical Cross Entropy loss function measures the difference between the predicted
    probability distribution ($\hat{y}_{i,j}$) and the actual one-hot encoded target distribution ($y_{i,j}$) over a
    batch of size $N$ with $C$ classes.

  - **$\hat{y}_{i,j}$**: This represents the predicted probability for the $j$-th class of the $i$-th example in the
    batch, clipped to avoid taking the log of zero.

  - **$y_{i,j}$**: This is the actual target probability for the $j$-th class of the $i$-th example in the batch. In
    one-hot encoding, this will be 1 for the correct class and 0 for all others.

  - **$\text{doutput}(i, j)$**: The derivative of the loss function with respect to the predicted probability for the
    $j$-th class of the $i$-th example. It represents the difference between the predicted probability and the actual
    target, guiding the backpropagation process.

- <code style="color: teal;">Factory Method</code> The LossFunction class includes a factory method, create, which
  instantiates a specific loss function object based on the provided LossType. This allows for flexibility in choosing
  the appropriate loss function during model creation.

---

### Optimizer

The Optimizer class and its derived classes implement various optimization algorithms used in training neural networks
to minimize the loss function and update the model parameters efficiently. The key functionalities of each class are as
follows:

- <code style="color: teal;">SGD</code> (Stochastic Gradient Descent) This optimizer updates weights and biases by
  applying a simple learning rate-scaled gradient descent step. It uses the applyUpdates method from the
  TensorOperations class to adjust the parameters directly with gradients.

  <h3 style="display: inline-block;">$$\text{weights} = \text{weights} - \eta \cdot \text{dweights}$$</h3>
  <h3 style="display: inline-block;">$$\text{biases} = \text{biases} - \eta \cdot \text{dbiases}$$</h3>

  - **$\text{weights}$**: The weights of the model that are being updated.
  - **$\text{dweights}$**: The gradient of the loss with respect to the weights, calculated during backpropagation.
  - **$\eta$**: The learning rate, a hyperparameter that controls the size of the update step.
  - **$\text{biases}$**: The biases of the model that are being updated.
  - **$\text{dbiases}$**: The gradient of the loss with respect to the biases, calculated during backpropagation.

- <code style="color: teal;">SGD with Momentum</code> This optimizer extends SGD by adding a momentum term that helps to
  accelerate the optimization process in the relevant direction and dampen oscillations. It maintains velocity terms
  (v_weights and v_biases) for weights and biases, which are updated iteratively and used to adjust the model
  parameters.

  <h3 style="display: inline-block;">$$v_{\text{weights}} = \mu \cdot v_{\text{weights}} + \eta \cdot \text{dweights}$$</h3>
  <h3 style="display: inline-block;">$$\text{weights} = \text{weights} - v_{\text{weights}}$$</h3>
  <h3 style="display: inline-block;">$$v_{\text{biases}} = \mu \cdot v_{\text{biases}} + \eta \cdot \text{dbiases}$$</h3>
  <h3 style="display: inline-block;">$$\text{biases} = \text{biases} - v_{\text{biases}}$$</h3>

  - **$v_{\text{weights}}$**: The velocity term for the weights, which accumulates the past gradients scaled by the
    momentum factor $\mu$.
  - **$v_{\text{biases}}$**: The velocity term for the biases, similar to $v_{\text{weights}}$, but for biases.
  - **$\mu$**: The momentum coefficient, which controls the influence of the past gradients on the current update.
  - **$\eta$**: The learning rate, which scales the contribution of the current gradient to the update.
  - **$\text{weights}$**: The weights of the model that are being updated.
  - **$\text{dweights}$**: The gradient of the loss with respect to the weights, calculated during backpropagation.
  - **$\text{biases}$**: The biases of the model that are being updated.
  - **$\text{dbiases}$**: The gradient of the loss with respect to the biases, calculated during backpropagation.

- <code style="color: teal;">Adam</code> This optimizer combines the advantages of both the AdaGrad and RMSProp
  algorithms, maintaining adaptive learning rates for each parameter by computing first (mean) and second (variance)
  moments of the gradients. It employs bias-correction terms to ensure unbiased estimates of these moments. Adam
  maintains separate moving averages of gradients (m_weights, m_biases) and squared gradients (v_weights, v_biases) for
  both weights and biases.

  <h3 style="display: inline-block;">$$m_{\text{weights}} = \beta_1 \cdot m_{\text{weights}} + (1 - \beta_1) \cdot \text{dweights}$$</h3>
  <h3 style="display: inline-block;">$$v_{\text{weights}} = \beta_2 \cdot v_{\text{weights}} + (1 - \beta_2) \cdot \text{dweights}^2$$</h3>
  <h3 style="display: inline-block;">$$\hat{m}_{\text{weights}} = \frac{m_{\text{weights}}}{1 - \beta_1^t}$$</h3>
  <h3 style="display: inline-block;">$$\hat{v}_{\text{weights}} = \frac{v_{\text{weights}}}{1 - \beta_2^t}$$</h3>
  <h3 style="display: inline-block;">$$\text{weights} = \text{weights} - \eta \cdot \frac{\hat{m}_{\text{weights}}}{\sqrt{\hat{v}_{\text{weights}}} + \epsilon}$$</h3>
  <h3 style="display: inline-block;">$$m_{\text{biases}} = \beta_1 \cdot m_{\text{biases}} + (1 - \beta_1) \cdot \text{dbiases}$$</h3>
  <h3 style="display: inline-block;">$$v_{\text{biases}} = \beta_2 \cdot v_{\text{biases}} + (1 - \beta_2) \cdot \text{dbiases}^2$$</h3>
  <h3 style="display: inline-block;">$$\hat{m}_{\text{biases}} = \frac{m_{\text{biases}}}{1 - \beta_1^t}$$</h3>
  <h3 style="display: inline-block;">$$\hat{v}_{\text{biases}} = \frac{v_{\text{biases}}}{1 - \beta_2^t}$$</h3>
  <h3 style="display: inline-block;">$$\text{biases} = \text{biases} - \eta \cdot \frac{\hat{m}_{\text{biases}}}{\sqrt{\hat{v}_{\text{biases}}} + \epsilon}$$</h3>

  - **$m_{\text{weights}}$**: The first moment estimate (mean of the gradients) for the weights, which accumulates the
    gradients with an exponential decay rate $\beta_1$.
  - **$v_{\text{weights}}$**: The second moment estimate (uncentered variance of the gradients) for the weights, which
    accumulates the squared gradients with an exponential decay rate $\beta_2$.
  - **$\hat{m}_{\text{weights}}$**: The bias-corrected first moment estimate for the weights, which adjusts
    $m_{\text{weights}}$ to account for the initialization bias.
  - **$\hat{v}_{\text{weights}}$**: The bias-corrected second moment estimate for the weights, which adjusts
    $v_{\text{weights}}$ to account for the initialization bias.
  - **$\text{weights}$**: The weights of the model that are being updated.
  - **$\text{dweights}$**: The gradient of the loss with respect to the weights, calculated during backpropagation.
  - **$m_{\text{biases}}$**: The first moment estimate for the biases.
  - **$v_{\text{biases}}$**: The second moment estimate for the biases.
  - **$\hat{m}_{\text{biases}}$**: The bias-corrected first moment estimate for the biases.
  - **$\hat{v}_{\text{biases}}$**: The bias-corrected second moment estimate for the biases.
  - **$\text{biases}$**: The biases of the model that are being updated.
  - **$\text{dbiases}$**: The gradient of the loss with respect to the biases, calculated during backpropagation.
  - **$\beta_1, \beta_2$**: Exponential decay rates for the moment estimates.
  - **$\eta$**: The learning rate, which scales the update.
  - **$\epsilon$**: A small constant added for numerical stability, preventing division by zero.
  - **$t$**: The time step or iteration count, which helps in correcting the bias in the moment estimates.

- <code style="color: teal;">RMSprop</code> This optimizer maintains an exponentially decaying average of past squared
  gradients to divide the gradient element-wise, which helps in dealing with the vanishing learning rate problem. It
  updates the weights and biases based on the root mean square of gradients.

  <h3 style="display: inline-block;">$$s_{\text{weights}} = \beta \cdot s_{\text{weights}} + (1 - \beta) \cdot \text{dweights}^2$$</h3>
  <h3 style="display: inline-block;">$$\text{weights} = \text{weights} - \eta \cdot \frac{\text{dweights}}{\sqrt{s_{\text{weights}}} + \epsilon}$$</h3>
  <h3 style="display: inline-block;">$$s_{\text{biases}} = \beta \cdot s_{\text{biases}} + (1 - \beta) \cdot \text{dbiases}^2$$</h3>
  <h3 style="display: inline-block;">$$\text{biases} = \text{biases} - \eta \cdot \frac{\text{dbiases}}{\sqrt{s_{\text{biases}}} + \epsilon}$$</h3>

  - **$s_{\text{weights}}$**: The running average of the squared gradients for the weights, updated using the
    exponential decay rate $\beta$.
  - **$s_{\text{biases}}$**: The running average of the squared gradients for the biases, similar to
    $s_{\text{weights}}$, but for biases.
  - **$\beta$**: The decay rate that controls the averaging of the squared gradients. It determines how much past
    gradients influence the current average.
  - **$\eta$**: The learning rate, which scales the update step for weights and biases.
  - **$\text{weights}$**: The weights of the model that are being updated.
  - **$\text{dweights}$**: The gradient of the loss with respect to the weights, calculated during backpropagation.
  - **$\text{biases}$**: The biases of the model that are being updated.
  - **$\text{dbiases}$**: The gradient of the loss with respect to the biases, calculated during backpropagation.
  - **$\epsilon$**: A small constant added for numerical stability to prevent division by zero in the denominator.

- <code style="color: teal;">Factory Method</code> The Optimizer class includes a factory method, create, which
  instantiates a specific optimizer object based on the provided OptimizerType and parameters. This allows for
  flexibility in choosing the appropriate optimizer during model training.

---

### Learning Decay

The `LearningDecay` class implements various learning rate decay strategies used to adjust the learning rate during
training, improving convergence for neural network models. The class allows different decay methods to be applied based
on the specified `LearningDecayType` and associated parameters:

- <code style="color: teal;">Decay Type Initialization</code> The constructor accepts a `LearningDecayType` and a map of
  parameters, setting default values if specific parameters are not provided. This ensures the decay method can operate
  with sensible defaults.

- <code style="color: teal;">Exponential Decay</code> Reduces the learning rate exponentially based on a specified decay
  rate. The learning rate decreases continuously over epochs, following the formula:
  <h3 style="display: inline-block;">$$\text{learningRrate} = \text{initialLearningRate} \times \text{decayRate}^{\text{epoch}}$$</h3>

- <code style="color: teal;">Step Decay</code> Reduces the learning rate at fixed intervals or "steps" by a decay
  factor. The learning rate is adjusted using the formula:
  <h3 style="display: inline-block;">$$\text{learningRate} = \text{initialLearningRate} \times \text{decayFactor}^{\left(\frac{\text{epoch}}{\text{stepSize}}\right)}$$</h3>

- <code style="color: teal;">Polynomial Decay</code> Reduces the learning rate following a polynomial function of the
  epoch, reaching a specified `end_learning_rate` after a certain number of `decay_steps`. The formula used is:
  <h3 style="display: inline-block;">$$\text{learningRate} = (\text{initialLearningRate} - \text{endLearningRate}) \times \left(1 - \frac{\text{epoch}}{\text{decaySteps}}\right)^{\text{power}} + \text{endLearningRate}$$</h3>

- <code style="color: teal;">Inverse Time Decay</code> Applies decay so that the learning rate decreases proportionally
  to the inverse of time, using the formula:
  <h3 style="display: inline-block;">$$\text{learningRate} = \frac{\text{initialLearningRate}}{1 + \text{decayRate} \times \frac{\text{epoch}}{\text{decaySteps}}}$$</h3>

- <code style="color: teal;">Cosine Decay</code> Applies a cosine decay function to reduce the learning rate, smoothly
  decreasing the learning rate and then increasing it slightly at the end. The formula used is:
  <h3 style="display: inline-block;">$$\text{learningRate} = \text{initialLearningRate} \times \frac{1 + \alpha \times \cos\left(\frac{\pi \times \text{epoch}}{\text{decaySteps}}\right)}{2}$$</h3>

- <code style="color: teal;">None</code> If no decay is specified, the learning rate remains constant throughout the
  training process.

## Getting Started

### Step 1: Clone the Repository

Begin by cloning the CNN-CPP project repository to your local machine. Use the following command in your terminal:

```bash
git clone https://github.com/kolemare/CNN-CPP.git
```

---

### Step 2: Installing Dependencies

### Linux

#### Option 1: Native Installation

For Linux users preferring a native setup without Docker, install the necessary dependencies with provided
**install.sh** script.

```bash
sudo ./install.sh
```

This script will install all required software on your Linux machine. This ensures that all necessary components are
properly set up for your development environment.

#### Option 2: Docker Installation

Linux docker setup: [Docker Desktop for Linux installation guide](https://docs.docker.com/engine/install/ubuntu/).

### Windows & Mac OS

Windows docker setup:
[Docker Desktop for Windows installation guide](https://docs.docker.com/desktop/install/windows-install/).  
Mac OS docker setup: [Docker Desktop for Mac installation guide](https://docs.docker.com/desktop/install/mac-install/).

---

### Step 3: Only For Docker

**Linux:** Before running Docker commands, make sure that your user belongs to the docker group to avoid using sudo with
every Docker command. If not already configured, you can add your user to the docker group with:

```bash
sudo usermod -aG docker $USER
```

**Windows:** Ensure that the Docker Desktop application is running before executing any Docker commands. This
application manages Docker engine and allows container management on Windows.

**Mac OS:** Docker on Mac behaves similarly to Linux. Ensure Docker Desktop is running as it provides the necessary
Docker engine to manage containers. You do not need to start a separate Docker service as with Windows.

First, ensure you are positioned at the root of the repository. The Docker image needs to be built only once. To build
the Docker image, use the following command:

```bash
docker build -t cnn-cpp-image .
```

To start the Docker container in interactive mode while mounting the repository directory, use:

```bash
docker run -it -v $(pwd):/CNN_CPP cnn-cpp-image
```

This command starts the Docker container in interactive mode and mounts the entire repository. This setup ensures that
any changes made within the Docker container are immediately visible on the host operating system, and vice versa. To
stop the running Docker container, you can use CTRL+C. Note that if the container is stopped, you will need to rerun the
command to start it again.

After you are done working, you can remove the Docker container to free up resources. Use the following commands to find
the container ID and remove it:

```bash
docker images  # Lists all containers, find the one you want to remove
```

```bash
docker rm [container_id]  # Replace [container_id] with your actual container ID
```

Once the Docker image is removed, you will need to rebuild it if you wish to use Docker again for the CNN-CPP Framework.
Building the Docker image is a one-time requirement unless it is deleted.  
This ensures that all necessary components are properly set up for your development environment.  
Size of Docker image for CNN-CPP is approximately: **1.85GB**

### Dataset Structure

### Graphs

### Image

- Loading
- Augmenting
- Resizing

## Model Example

## Results
