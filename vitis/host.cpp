#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#define DATA_SIZE 4096

#include <vector>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <CL/cl2.hpp>
#include "ap_int.h"

#include "mmkernel.h"

typedef ap_int<16> dtype;

//mmkernel defined constants
#define CIN 64
#define STRIDE 1
#define K_H 1
#define K_W 1
#define K_BOUND K_H*K_W
#define COUT 3
#define INMAP_H 16
#define INMAP_W 16
#define FMAPO_H INMAP_H/STRIDE
#define FMAPO_W INMAP_W/STRIDE
#define O_2 FMAPO_H*FMAPO_W

// hardware parameters: cannot be changed for different kernel calls
// UF: unroll factor
#define COUT_UF 32
#define FMAP_UF 32
// #define Wdepth Cout/Cout_UF*Cin*K_H*K_W

// hyper-params for winograd
#define m 2
#define r 3
#define Inmap_H_wino INMAP_H/m
#define Inmap_W_wino INMAP_W/m

// Forward declaration of utility functions included at the end of this file
std::vector<cl::Device> get_xilinx_devices();
char *read_binary_file(const std::string &xclbin_file_name, unsigned &nb);

// ------------------------------------------------------------------------------------
// Main program
// ------------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    // ------------------------------------------------------------------------------------
    // Step 1: Initialize the OpenCL environment
    // ------------------------------------------------------------------------------------
    cl_int err;
    std::string binaryFile = (argc != 2) ? "mmkernel.xclbin" : argv[1];
    unsigned fileBufSize;
    std::vector<cl::Device> devices = get_xilinx_devices();
    devices.resize(1);
    cl::Device device = devices[0];
    cl::Context context(device, NULL, NULL, NULL, &err);
    char *fileBuf = read_binary_file(binaryFile, fileBufSize);
    cl::Program::Binaries bins{{fileBuf, fileBufSize}};
    cl::Program program(context, devices, bins, NULL, &err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    cl::Kernel krnl_matrix_mult(program, "top_kernel", &err);

    // ------------------------------------------------------------------------------------
    // Step 2: Create buffers and initialize test values
    // ------------------------------------------------------------------------------------
    // Create the buffers and allocate memory
    cl::Buffer inMap_buf(context, CL_MEM_READ_ONLY, sizeof(dtype) * INMAP_H * INMAP_W, NULL, &err);
    cl::Buffer knl_buf(context, CL_MEM_READ_ONLY, sizeof(dtype) * K_H * K_W * CIN * COUT, NULL, &err);
    cl::Buffer outMap_buf(context, CL_MEM_WRITE_ONLY, sizeof(dtype) * FMAPO_H * FMAPO_W * COUT, NULL, &err);

    // Map buffers to kernel arguments, thereby assigning them to specific device memory banks
    krnl_matrix_mult.setArg(0, inMap_buf);
    krnl_matrix_mult.setArg(1, knl_buf);
    krnl_matrix_mult.setArg(2, outMap_buf);


    // Map host-side buffer memory to user-space pointers
    dtype *inMap = (dtype *)q.enqueueMapBuffer(inMap_buf, CL_TRUE, CL_MAP_WRITE, 0, sizeof(dtype) * INMAP_H * INMAP_W);
    dtype *knl = (dtype *)q.enqueueMapBuffer(knl_buf, CL_TRUE, CL_MAP_WRITE, 0, sizeof(dtype) * K_H * K_W * CIN * COUT);
    dtype *outMap = (dtype *)q.enqueueMapBuffer(outMap_buf, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, sizeof(dtype) * FMAPO_H * FMAPO_W * COUT);

    // Initialize the vectors used in the test
    for (int i=0; i< INMAP_H * INMAP_W; ++i){
        inMap[i] = i % (INMAP_H * INMAP_W);
    }

    int count = 0;
    std::cout << "---------------\nInput Feature Map\n";
    for (int i=0; i<INMAP_H; ++i){
        for (int j=0; j<INMAP_W; ++j){
            std::cout << inMap[count] << " ";
            ++count;
        }
        std::cout << "\n";
    }

    for (int i=0; i<K_H * K_W * CIN * COUT; ++i){
        knl[i] = 1;
    }

    count = 0;
    std::cout << "---------------\nKernel\n";
    for (int i=0; i<COUT*CIN; ++i){
        std::cout << "Kernel " << i << "\n";
        for (int j=0; j<K_H; ++j){
            for (int k=0; k<K_W; ++k){
                // std::cout << knl[count] << " ";
                // ++count;
            }
            std::cout << "\n";
        }
    }

    for (int i=0; i<FMAPO_H * FMAPO_W * COUT; ++i){
        outMap[i] = 0;
    }


    // ------------------------------------------------------------------------------------
    // Step 3: Run the kernel
    // ------------------------------------------------------------------------------------
    // Set kernel arguments
    krnl_matrix_mult.setArg(0, inMap_buf);
    krnl_matrix_mult.setArg(1, knl_buf);
    krnl_matrix_mult.setArg(2, outMap_buf);
    krnl_matrix_mult.setArg(3, 0); //alg_last
    krnl_matrix_mult.setArg(4, 0); //alg_current
    krnl_matrix_mult.setArg(5, 0); //alg_next


    // Schedule transfer of inputs to device memory, execution of kernel, and transfer of outputs back to host memory
    q.enqueueMigrateMemObjects({inMap_buf, knl_buf}, 0 /* 0 means from host*/);
    q.enqueueTask(krnl_matrix_mult);
    q.enqueueMigrateMemObjects({outMap_buf}, CL_MIGRATE_MEM_OBJECT_HOST);

    // Wait for all scheduled operations to finish
    q.finish();

    // ------------------------------------------------------------------------------------
    // Step 4: Check Results and Release Allocated Resources
    // ------------------------------------------------------------------------------------
    bool match = false;
    // for (int i = 0; i < FMAPO_H * FMAPO_W ; i++)
    // {
    //     int expected = in1[i] + in2[i];
    //     if (out[i] != expected)
    //     {
    //         std::cout << "Error: Result mismatch" << std::endl;
    //         std::cout << "i = " << i << " CPU result = " << expected << " Device result = " << out[i] << std::endl;
    //         match = false;
    //         break;
    //     }
    // }
    std::cout << "--------------\nOutput Feature Map\n";
    count = 0;
    for (int i=0; i<COUT; ++i){
        std::cout << "OFM " << i << "\n";
        for (int j=0; j<FMAPO_H; ++j){
            for (int k=0; k<FMAPO_W; ++k){
                std::cout << outMap[count] << " ";
                ++count;
            }
            std::cout << "\n";
        }
    }

    delete[] fileBuf;

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}

// ------------------------------------------------------------------------------------
// Utility functions
// ------------------------------------------------------------------------------------
std::vector<cl::Device> get_xilinx_devices()
{
    size_t i;
    cl_int err;
    std::vector<cl::Platform> platforms;
    err = cl::Platform::get(&platforms);
    cl::Platform platform;
    for (i = 0; i < platforms.size(); i++)
    {
        platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(&err);
        if (platformName == "Xilinx")
        {
            std::cout << "INFO: Found Xilinx Platform" << std::endl;
            break;
        }
    }
    if (i == platforms.size())
    {
        std::cout << "ERROR: Failed to find Xilinx platform" << std::endl;
        exit(EXIT_FAILURE);
    }

    //Getting ACCELERATOR Devices and selecting 1st such device
    std::vector<cl::Device> devices;
    err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
    return devices;
}

char *read_binary_file(const std::string &xclbin_file_name, unsigned &nb)
{
    if (access(xclbin_file_name.c_str(), R_OK) != 0)
    {
        printf("ERROR: %s xclbin not available please build\n", xclbin_file_name.c_str());
        exit(EXIT_FAILURE);
    }
    //Loading XCL Bin into char buffer
    std::cout << "INFO: Loading '" << xclbin_file_name << "'\n";
    std::ifstream bin_file(xclbin_file_name.c_str(), std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    char *buf = new char[nb];
    bin_file.read(buf, nb);
    return buf;
}