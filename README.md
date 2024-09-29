# OpenCL Workspace

## Instructions
```
sudo apt install opencl-headers ocl-icd-opencl-dev -y    
sudo apt install libspdlog-dev 
```

## Usage
```
./build/gemm <M> <N> <K> <path to kernel>
```
where M, N, K are matrix dimensions