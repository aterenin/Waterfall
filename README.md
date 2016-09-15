# Waterfall

A Scala DSL prividing a medium-level API for fast, simple, readable numerical computing on GPUs.

### Motivation

CUDA is lightning fast, but it's a little bit ridiculous when you have to write the following to make a draw from a multivariate normal distribution with mean vector mu and precision matrix Psi.

```C
CUSOLVER_CALL(cusolverDnSpotrf(cusolverHandle, CUBLAS_FILL_MODE_UPPER, p, Psi, p, cholWorkspace, cholWorkspaceNumBytes, cusolverDevInfo));
CURAND_CALL(curandGenerateNormal(curandGenerator, beta, p, 0.0, 1.0));
CUBLAS_CALL(cublasStrsv(cublasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, p, Psi, p, beta, 1));
CUBLAS_CALL(cublasSaxpy(cublasHandle, p, ptrOnef, mu, 1, beta, 1));
```

If you have no idea what the above code is doing, then Waterfall is for you. The exact same computation as above can be written as follows - in this case, with effectively zero performance cost.

```Scala
R =: Psi.computeCholesky(workspace)
beta =: Random.normal
beta =: R.inv * beta
beta =: beta + mu
```

This is what Waterfall is all about. Written in Scala, Waterfall uses many modern programming features to make GPU-accelerated numerical computing as painless as possible, while providing excellent performance by maintaining a 1:1 mapping to the underlying CUDA calls. 

Waterfall was created after I (@aterenin) came back presenting my work on [GPU-accelerated Gibbs Sampling](http://arxiv.org/abs/1608.04329) at [MCQMC](http://mcqmc2016.stanford.edu), and was unable to understand what my own CUDA code was doing after not seeing it for a week.

### Getting Started

Waterfall is currently in alpha testing. To get started, do the following.

  1. Clone the repository, and import the project into your favorite Scala IDE (we recommend IntelliJ IDEA).
  2. Make a new folder called `lib` inside of your project directory, download `JCuda`, and place its jars and platform-specific files into the `lib` folder.
  3. Add the `lib` folder to the PATH variable in your editor.
  
Now, you can write your app. Be sure to call `Waterfall.init()` at the beginning of your code.

### Design, Philosophy, and Usage

Waterfall is a Scala DSL for numerical computing on GPUs. It is designed to balance user friendliness with performance: its syntax tries best to resemble mathematical formulae, yet every single operation corresponds exactly to a low-level CUDA call for maximum performance. It tries best to guide the programmer toward doing the right thing: for instance, it is equivalent and much faster to solve a linear system than to compute and multiply an inverse matrix - Waterfall intentionally makes the former easier to do than the latter.

The basic classes in Waterfall are `GPUMatrix`, `GPUVector`, and `GPUConstant`. All are backed by column-major floating-point arrays on the GPU. The fundamental notion in Waterfall is that of a *computation* - a CUDA routine that takes an input, and can be stored in some output. A computation is created by performing operations on classes. It is not evaluated immediately, and is executed through the assignment operator `=:`.

This can be illustrated by example. First, let's create two input matrices `X` and `Y`, and an output matrix `Z`.

```Scala
val X = GPUMatrix.createFromColumnMajorArray(hostX)
val Y = GPUMatrix.createFromColumnMajorArray(hostY)
val Z = GPUMatrix.create(X.numRows, Y.numCols)
```

Matrices and vectors know their parameters: calling `X.numRows` and `Y.numCols` yields the expected behavior. A matrix can be copied from the GPU to the host by doing the following.

```Scala
val hostX = X.copyToHost
```

Now, let's multiply `X` by `Y` and store the result in `Z`
  
```Scala
Z =: X * Y
```

Internally, the operation `X * Y` causes Waterfall to create a `GPUMatrixResult` class. Then, the operation `Z =: ...` causes the computation to be executed and stored in `Z`. Waterfall will check `X` and `Y` for compatibility and will try its best to provide meaningful messages if an exception is encountered.

Result classes themselves cannot be multiplied, because this would require executing more than one computation. Thus, the following is invalid.

```Scala
W =: X * Y * Z // won't compile
```

This makes Waterfall a lower-level framework than typical numerical computing languages such as *R*. This has its downsides: large formulae must be written as sequences of smaller ones. It also has its upsides: it makes clearly obvious how everything is being calculated, and what buffer space is needed along the way, simplifying optimization.

Once a matrix is created, it can be declared symmetric or triangular. Calling `X.declareSymmetric` will return a `SymmetricMatrix`, for which multiplication is strictly faster than for general matrices. This may also open up new computations: symmetric matrices can be decomposed into Cholesky factors.

Waterfall evaluates matrix transposition *lazily*: calling `X.T` returns a `GPUMatrix` that can be used immediately for any given purpose, and does not perform any computations on the GPU. Let's illustrate this.

```Scala
val XT = X.T
```
No evaluation has taken place on the GPU - instead, `XT` contains a `GPUMatrix` object that internally knows that it has been transposed, which it will use the next time it needs to, such as when it is multiplied by another matrix. This immediately yields better performance compared to other languages such as *R* where transposes are evaluated eagerly.

Matrices can have constants lazily attached to them via the `GPUMatrix.withConstant` method. Constants can be consumed in some computations, based on whatever is supported in the underlying CUDA routines.

Matrix inversion is also performed lazily. Currently, symmetric and triangular matrix inversion are supported. Triangular matrices can be inverted immediately, symmetric matrices must have an attached Cholesky decomposition computed. Let's see an example: suppose that `X` is a `SymmetricMatrix`, and `w` and `v` are vectors.

```Scala
val R = GPUMatrix.create(X.size,X.size).declareTriangular
val workspace = createCholeskyWorkspace(X)
R =: X.computeCholesky(workspace)
w = X.inv * v
``` 

The above will compute a Cholesky decomposition of `X` and store it in `R`. The `workspace` class contains the `buffer` and `devinfo` parameters needed by CUDA to perform a Cholesky factorization. Once `X` can be inverted, calling `X.inv` will return an `InverseSymmetricMatrix` which can be multiplied by matrices and vectors. Multiplying an inverse matrix by a matrix or vector will create a computation in which the equivalent linear system is solved - strictly faster and more numerically stable than calculating the inverse directly. Inverse matrices cannot be added: if this is truly necessary, the user should multiply the `InverseMatrix` by the identity to force evaluation.

Every single computation in Waterfall corresponds directly to a CUDA routine. For example, the computation `Z =: X * Y` where `X` is symmetric corresponds to `cublasSsymm`, and `R =: X.computeCholesky(workspace)` corresponds to `cusolverSpotrf`.

Waterfall includes a `Random` object capable of generating arrays of independent Gaussians and independent Uniforms. Its use is analogous to the rest of the package. 

```Scala
X =: Random.normal
``` 

The above will execute as expected.

Waterfall supports *CUDA streams*. They can be created and used as follows.

```Scala
val stream = Stream.create
{
  implicit val s = stream
  // all code here will now execute in stream s
}
```
Any Waterfall operations performed inside the code block will execute in the stream. Streams can be synchronized in a blocking fashion via `stream.synchronize`.

Waterfall makes it easy to load and execute custom CUDA kernels written by the user in CUDA C. To do so, it suffices to do the following.

```Scala
val customKernel = CustomKernelFile("kernel.ptx").loadCustomKernel("custom_kernel")
customKernel(gridDimX, blockDimX)(arg1, arg2, arg3)
```

The above will load the kernel `custom_kernel` from the file `kernel.ptx`. Then, it will launch the kernel with grid and block dimensions `(gridDimX, blockDimX)` and arguments `(arg1, arg2, arg3)`. The API supports the same launch configuration as CUDA, as well as shared memory parameters, and arbitrary arguments. All Waterfall classes such as `GPUMatrix` and `GPUVector` can be passed as arguments immediately and will internally be converted as needed to yield expected behavior.

Waterfall exposes its internals. All of its classes expose the underlying `Pointer` classes used by JCuda, which can be used for any purpose necessary.

Waterfall *never* allocates memory internally. All memory is allocated by the user when creating matrices and vectors - this simplifies optimization. Some operations may perform a memory copy, but most will not. Some operations support in-place mode - operations that don't will throw an exception.

Waterfall classes are *almost* immutable. They can mutate only through the assignment operator `=:`, which might change the transpose flag so that it reflects reality, attach matrix decompositions, or other minor needed changes. The underlying `Pointer` classes attached cannot change.

For maximum performance on GPUs, at the moment Waterfall operates exclusively in floating point precision.

Waterfall checks all of its operations for compatibiliy, and also checks all of the return values given by CUDA routines. This simplifies debugging.

Waterfall is under active development, and strives to be fully unit tested - coverage is currently close but not quite fully complete. If you have found a bug, please submit an issue. We hope that this DSL is useful, and would love your feedback - please feel free to get in touch by email at `{my-github-username}@ucsc.edu`.