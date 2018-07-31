# hmglib - *H*ierarchical *m*atrices on *G*PU(s) *lib*rary
This library provides an entirely GPU-based implementation of the *hierarchical (H) matrix* approach. H matrices approximate specific dense matrices, e.g., from discretized integral equations or kernel ridge regression, leading to log-linear time complexity in dense matrix-vector products. The library specifically allows to construct and apply an H matrix on GPU(s). To get high performance on GPU(s), techniques like parallel tree traversal, space filling curves and batched operations are applied.

## Getting Started

After cloning the repository, go to the source directory and adapt the [makefile](src/Makefile) to your needs.

### Prerequisites
The following list of libraries should allow you to compile the example codes.

- Nvidia GPU with CUDA support
- recent version of the CUDA Toolkit
- recent version of OpenBLAS
- recent version of MAGMA
- recent version of GSL (GNU Scientific Library)

### Compilation of Example Code 

After appropriately modifying the [makefile](src/Makefile), go to the source directory and compile the code by running
```
make
```

### Example Code

The file [hmglib_test.cu](src/hmglib_test.cu) is an example code that allows to construct an H matrix for a dense matrix from kernel ridge regression with a Gaussian kernel and random sampling points. After the construction of the H matrix, it runs the H matrix - vector product and compares the solution of this product to a full matrix - vector product. Finally, the error of this produc is reported.

#### Running the Example Code

Run the example code by entering
```
./hmglib_test <Nx> <Ny> <k> <c_leaf> <exponent of epsilon> <eta>
```

A typical (small!) example would be:
```
./hmglib_test 4096 4096 16 256 -3 1.0
```

Here, a matrix of 4096 x 4096 is approximated with adaptive cross approximation (ACA) with 16 low-rank terms, a leaf size of 256 and a balancing parameter of 1.0. The *exponent of epsilon* is currently unused.

### How to go further ?

It is first suggested to read reference [1], found in the literature section of this page. To construct H matrices, for other matrix types than a Gaussian kernel matrix, an interested user should try to derive a new *system assembler* class, providing a method to evaluate a given entry of the (dense) matrix. In the example code from above, the implemented class is called *gaussian_kernel_system_assembler*.

## Current State of the Project

The current state of the software project is that it is an experimental code, which was used to create the two manuscripts referenced below. *hmglib* is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License in [LICENSE](LICENSE) for more details.

## Authors

* **Peter Zaspel** - *Main developer* - [University of Basel](https://dmi.unibas.ch/de/personen/peter-zaspel/)

## License

This project is licensed under the LGPL License version 3.0 - see the [LICENSE](LICENSE) file for details.

## Literature
1. P. Zaspel. Algorithmic patterns for H matrices on many-core processors, Preprint 2017-12, Fachbereich Mathematik, Universität Basel, Switzerland, 2017. Also available as arXiv preprint [arXiv:1708.09707](https://arxiv.org/abs/1708.09707).
2. H. Harbrecht and P. Zaspel. A scalable H-matrix approach for the solution of boundary integral equations on multi-GPU clusters. Preprint 2018-11, Fachbereich Mathematik, Universität Basel, Switzerland, 2018. Available as arXiv preprint [arXiv:1806.11558](https://arxiv.org/abs/1806.11558).

