## About
Here, I provide my implementation for my paper "[The effect of the number of parameters and the number of local feature patches on loss landscapes in distributed quantum neural networks](https://arxiv.org/abs/2504.19239)". 

## Repository Structure
* **`src/`**: Contains all the source code for training our models.
* **`src_hessian_analysis/`**: Contains all the source codes for computing the largest eigenvalue of Hessian. 
    * **`src_hessian_analysis/requirements_for_gh200.txt`**: Lists major Python dependencies requirred to run codes using NVIDIA GH200 at Lambdalabs' cloud service. 
    * **`src_hessian_analysis/src_miyabi/`**: Includes scripts, requirements.txt, and definition file for environment setup to use supercomputer Miyabi, i.e., the Supermicro ARS-111GL-DNHR-LCC and NVIDIA Hopper H100 GPU (Miyabi-G) at Joint Center for Advanced High Performance Computing (JCAHPC). 
* **`src_vis/`**: Includes source codes for computing losses to visualize loss surfaces. 
* **`requirements.txt`**: Lists major Python dependencies required to run our codes, ensuring reproducibility. 
* **`README.md`**: This file, providing an overview and instructions for this repository.

## Citation
If you find this repository useful for your research, please consider citing this work:
```
@article{kawase2025effect,
  title={The effect of the number of parameters and the number of local feature patches on loss landscapes in distributed quantum neural networks},
  author={Kawase, Yoshiaki},
  journal={arXiv preprint arXiv:2504.19239},
  year={2025}
}
```