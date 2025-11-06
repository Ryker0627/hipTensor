#include <hiptensor/hiptensor.h>
#include <hiptensor/hiptensor_types.h>
#include <numeric>
#include <unordered_map>

#include "common.hpp"

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          hiptensorDataType_t          typeA,
          hiptensorDataType_t          typeB,
          hiptensorDataType_t          typeC,
          hiptensorComputeDescriptor_t typeCompute>
int bilinearContraction(void* alpha, void* beta, std::vector<int> &modeA,
                              std::vector<int> &modeB, std::vector<int> &modeC, std::unordered_map<int, int64_t> &extent)
{
    //std::vector<int> modeC{'m', 'n', 'u', 'v'};
    //std::vector<int> modeA{'m', 'n', 'h', 'k'};
    //std::vector<int> modeB{'u', 'v', 'h', 'k'};

    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    //std::unordered_map<int, int64_t> extent;

    //extent['m'] = 256;
    //extent['n'] = 20;
    //extent['u'] = 128;
    //extent['v'] = 128;
    //extent['h'] = 64;
    //extent['k'] = 64;

    std::vector<int64_t> c_ms_ns_lengths;
    //std::cout<<"modeC"<<std::endl;
    for(auto mode : modeC)
    {
        c_ms_ns_lengths.push_back(extent[mode]);
        //std::cout<<mode<<" "<<extent[mode]<<std::endl;
    }
    //std::cout<<std::endl;

    std::vector<int64_t> a_ms_ks_lengths;
    //std::cout<<"modeA"<<std::endl;
    for(auto mode : modeA)
    {
        a_ms_ks_lengths.push_back(extent[mode]);
        //std::cout<<mode<<" "<<extent[mode]<<std::endl;
    }

    std::vector<int64_t> b_ns_ks_lengths;
    //std::cout<<"modeB"<<std::endl;
    for(auto mode : modeB)
    {
        b_ns_ks_lengths.push_back(extent[mode]);
        //std::cout<<mode<<" "<<extent[mode]<<std::endl;
    }

    /**********************
     * Allocating data
     **********************/
    std::cout << "Initializing host data..." << std::endl;

    size_t elementsA = std::accumulate(
        a_ms_ks_lengths.begin(), a_ms_ks_lengths.end(), size_t{1}, std::multiplies<size_t>());
    size_t elementsB = std::accumulate(
        b_ns_ks_lengths.begin(), b_ns_ks_lengths.end(), size_t{1}, std::multiplies<size_t>());
    size_t elementsC = std::accumulate(
        c_ms_ns_lengths.begin(), c_ms_ns_lengths.end(), size_t{1}, std::multiplies<size_t>());

    size_t sizeA = sizeof(ADataType) * elementsA;
    size_t sizeB = sizeof(BDataType) * elementsB;
    size_t sizeC = sizeof(CDataType) * elementsC;

    ADataType* A = nullptr;
    BDataType* B = nullptr;
    CDataType* C = nullptr;
    CHECK_HIP_ERROR(hipHostMalloc((void**)&A, sizeA));
    CHECK_HIP_ERROR(hipHostMalloc((void**)&B, sizeB));
    CHECK_HIP_ERROR(hipHostMalloc((void**)&C, sizeC));

    void *A_d, *B_d, *C_d;

    CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&A_d), sizeA));
    CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&B_d), sizeB));
    CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&C_d), sizeC));

    /*******************
     * Initialize data
     *******************/
    int initMethod = 1; // TODO read value from commandline
    for(int64_t i = 0; i < elementsA; i++)
    {
        if(initMethod == 0)
        {
            A[i] = ADataType(float(std::rand()) / float(RAND_MAX) - 0.5) * 100;
        }
        else
        {
            A[i] = (ADataType)(float(i) / 100);
        }
    }

    for(int64_t i = 0; i < elementsB; i++)
    {
        if(initMethod == 0)
        {
            B[i] = BDataType(float(std::rand()) / float(RAND_MAX) - 0.5) * 100;
        }
        else
        {
            B[i] = (BDataType)(float(i) / 100);
        }
    }

    for(int64_t i = 0; i < elementsC; i++)
    {
        if(initMethod == 0)
        {
            C[i] = CDataType(float(std::rand()) / float(RAND_MAX) - 0.5) * 100;
        }
        else
        {
            C[i] = (BDataType)(float(i) / 100);
        }
    }

    /********************************************
     * Transfer the Host Tensor to Device Memory
     ********************************************/
    std::cout << "Initializing device data..." << std::endl;

    CHECK_HIP_ERROR(hipMemcpy(A_d, static_cast<const void*>(A), sizeA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(B_d, static_cast<const void*>(B), sizeB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(C_d, static_cast<const void*>(C), sizeC, hipMemcpyHostToDevice));

    /************************************************
     * Retrieve the memory alignment for each tensor
     ************************************************/
    uint32_t          alignmentRequirement = 1;
    hiptensorHandle_t handle;
    CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));

    CHECK_HIPTENSOR_ERROR(hiptensorLoggerSetMask(HIPTENSOR_LOG_LEVEL_PERF_TRACE));

    /********************************************
     * Initialize tensors with the input lengths
     ********************************************/
    hiptensorTensorDescriptor_t a_ms_ks = nullptr;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                          &a_ms_ks,
                                                          nmodeA,
                                                          a_ms_ks_lengths.data(),
                                                          NULL, /*stride*/
                                                          typeA,
                                                          alignmentRequirement));

    hiptensorTensorDescriptor_t b_ns_ks = nullptr;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                          &b_ns_ks,
                                                          nmodeB,
                                                          b_ns_ks_lengths.data(),
                                                          NULL, /*stride*/
                                                          typeB,
                                                          alignmentRequirement));

    hiptensorTensorDescriptor_t c_ms_ns = nullptr;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                          &c_ms_ns,
                                                          nmodeC,
                                                          c_ms_ns_lengths.data(),
                                                          NULL, /*stride*/
                                                          typeC,
                                                          alignmentRequirement));

    /********************************
     * Create Contraction Descriptor
     ********************************/

    hiptensorOperationDescriptor_t desc;
    CHECK_HIPTENSOR_ERROR(hiptensorCreateContraction(handle,
                                                     &desc,
                                                     a_ms_ks,
                                                     modeA.data(),
                                                     HIPTENSOR_OP_IDENTITY,
                                                     b_ns_ks,
                                                     modeB.data(),
                                                     HIPTENSOR_OP_IDENTITY,
                                                     c_ms_ns,
                                                     modeC.data(),
                                                     HIPTENSOR_OP_IDENTITY,
                                                     c_ms_ns,
                                                     modeC.data(),
                                                     typeCompute));

    /***************************
     * Set the algorithm to use
     ***************************/
    hiptensorPlanPreference_t planPref;
    CHECK_HIPTENSOR_ERROR(hiptensorCreatePlanPreference(
        handle, &planPref, HIPTENSOR_ALGO_DEFAULT /*HIPTENSOR_ALGO_ACTOR_CRITIC*/, HIPTENSOR_JIT_MODE_NONE));

    /**********************
     * Query workspace
     **********************/
    uint64_t worksize = 0;
    CHECK_HIPTENSOR_ERROR(hiptensorEstimateWorkspaceSize(
        handle, desc, planPref, HIPTENSOR_WORKSPACE_DEFAULT, &worksize));

    /**************************
     * Create Contraction Plan
     **************************/
    std::cout << "Initializing contraction plan..." << std::endl;

    hiptensorPlan_t plan;
    CHECK_HIPTENSOR_ERROR(hiptensorCreatePlan(handle, &plan, desc, planPref, worksize));

    // TODO query actually used workspace
    void* workspace = nullptr;

    if(worksize > 0)
    {
        CHECK_HIP_ERROR(hipMalloc(static_cast<void**>(&workspace), worksize));
    }

    std::cout << "Launching contraction kernel..." << std::endl;

    CHECK_HIPTENSOR_ERROR(hiptensorContract(
        handle, plan, alpha, A_d, B_d, beta, C_d, C_d, workspace, worksize, 0 /* stream */));

    CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
    CHECK_HIPTENSOR_ERROR(hiptensorDestroyPlanPreference(planPref));
    CHECK_HIPTENSOR_ERROR(hiptensorDestroyPlan(plan));
    CHECK_HIPTENSOR_ERROR(hiptensorDestroyOperationDescriptor(desc));
    if(a_ms_ks)
    {
        hiptensorDestroyTensorDescriptor(a_ms_ks);
        a_ms_ks = nullptr;
    }
    if(b_ns_ks)
    {
        hiptensorDestroyTensorDescriptor(b_ns_ks);
        b_ns_ks = nullptr;
    }
    if(c_ms_ns)
    {
        hiptensorDestroyTensorDescriptor(c_ms_ns);
        c_ms_ns = nullptr;
    }

    HIPTENSOR_FREE_HOST(A);
    HIPTENSOR_FREE_HOST(B);
    HIPTENSOR_FREE_HOST(C);

    HIPTENSOR_FREE_DEVICE(A_d);
    HIPTENSOR_FREE_DEVICE(B_d);
    HIPTENSOR_FREE_DEVICE(C_d);
    HIPTENSOR_FREE_DEVICE(workspace);

    return 0;
}

std::vector<int> stringToVectorInt(const std::string& strNumbers)
{
    std::vector<int> numbers;
    std::stringstream ss(strNumbers);
    std::string item;

    while(std::getline(ss, item, ','))
    {
        if(!item.empty()) numbers.push_back(stoi(item));
    }

    return numbers;
}

void getExtent(const std::vector<int> &mode, const std::vector<int> &lens, std::unordered_map<int,int64_t> &extent)
{
    for(int i=0; i<mode.size(); i++)
    {
       extent[mode[i]] = lens[i];
    }
}

void printVector(const std::vector<int> &arr)
{
    std::cout<<"[";
    for(int i=0; i<arr.size(); i++) {
        std::cout<<arr[i];
        if(i<arr.size()-1) std::cout<<", ";
    }
    std::cout<<"]";
}

std::vector<int> getLengths(const std::vector<int> &mode, const std::unordered_map<int,int64_t> &extent) {
    std::vector<int> lengths;
    for(auto imode:mode) lengths.push_back(extent.at(imode));
    return lengths;
}

void printRunHead(const std::string strTitle, const std::vector<int> &modeA, const std::vector<int> &modeB,
                  const std::vector<int> &modeC, const std::unordered_map<int,int64_t> &extent)
{
        std::cout << "\n\nData Type: "<< strTitle << std::endl;
        std::cout<<"modes: ";
        printVector(modeA);
        printVector(modeB);
        printVector(modeC);
        std::cout <<"\nlengths: ";
        std::vector<int> lengths = getLengths(modeA, extent);
        printVector(lengths);
        lengths = getLengths(modeB, extent);
        printVector(lengths);
        lengths = getLengths(modeC, extent);
        printVector(lengths);
        std::cout<<"\n====================================================="<<std::endl;
}

int main()
{
    std::vector<std::string> strModeA = {"0, 1, 4, 5", "0, 1, 2, 6, 7, 8", "0, 1, 2, 3, 8, 9, 10, 11",
                                         "0, 1, 2, 3, 8, 9, 10, 11", "0, 1, 2, 3, 4, 10, 11, 12, 13, 14",
                                        "0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17"};
    std::vector<std::string> strModeB = {"2, 3, 4, 5", "3, 4, 5, 6, 7, 8", "4, 5, 6, 7, 8, 9, 10, 11",
                                        "4, 5, 6, 7, 8, 9, 10, 11", "5, 6, 7, 8, 9, 10, 11, 12, 13, 14",
                                        "6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17"};
    std::vector<std::string> strModeC = {"0, 1, 2, 3", "0, 1, 2, 3, 4, 5", "0, 1, 2, 3, 4, 5, 6, 7",
                                        "0, 1, 2, 3, 4, 5, 6, 7", "0, 1, 2, 3, 4, 5, 6, 7, 8, 9",
                                        "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11"};

    std::vector<std::vector<int>> modesA;
    for(auto str:strModeA) modesA.push_back(stringToVectorInt(str));
    std::vector<std::vector<int>> modesB;
    for(auto str:strModeB) modesB.push_back(stringToVectorInt(str));
    std::vector<std::vector<int>> modesC;
    for(auto str:strModeC) modesC.push_back(stringToVectorInt(str));


    std::vector<std::vector<int>> lensA={{256, 20, 128, 128}, {8, 32, 32, 32, 32, 16}, {10, 8, 8, 8, 16, 16, 16, 2},
                                        {10, 8, 8, 8, 16, 16, 16, 2}, {16, 16, 2, 2, 8, 16, 16, 2, 8, 8},
                                        {16, 16, 2, 2, 2, 2, 16, 16, 2, 2, 8, 8}};
    std::vector<std::vector<int>> lensB={{128, 128, 128, 128},{32, 16, 32, 32, 32, 16}, {16, 16, 16, 8, 16, 16, 16, 2},
                                        {16, 16, 16, 8, 16, 16, 16, 2}, {8, 16, 2, 2, 8, 16, 16, 2, 8, 8},
                                        {16, 16, 2, 2, 2, 2, 16, 16, 2, 2, 8, 8}};
    std::vector<std::vector<int>> lensC={{256, 20, 128, 128}, {8, 32, 32, 32, 16, 32}, {10, 8, 8, 8, 16, 16, 16, 8},
                                        {10, 8, 8, 8, 16, 16, 16, 8}, {16, 16, 2, 2, 8, 8, 16, 2, 2, 8},
                                        {16, 16, 2, 2, 2, 2, 16, 16, 2, 2, 2, 2}};

    std::vector<std::unordered_map<int,int64_t>> extents;
    for(int i=0; i<modesA.size(); i++)
    {
        std::unordered_map<int,int64_t> extent;
        getExtent(modesA[i], lensA[i], extent);
        getExtent(modesB[i], lensB[i], extent);

        extents.push_back(extent);
    }

    typedef hip_bfloat16 DataType_bf16;
    typedef _Float16     DataType_f16;
    typedef float        DataType_f32;
    typedef double       DataType_f64;

    typedef float        floatTypeCompute;
    typedef hip_bfloat16 bf16TypeCompute;
    typedef _Float16     f16TypeCompute;
    typedef double       doubleTypeCompute;

    constexpr hiptensorDataType_t   type_16BF = HIPTENSOR_R_16BF;
    constexpr hiptensorDataType_t   type_16F  = HIPTENSOR_R_16F;
    constexpr hiptensorDataType_t   type_32F  = HIPTENSOR_R_32F;
    constexpr hiptensorDataType_t   type_64F  = HIPTENSOR_R_64F;

    constexpr hiptensorComputeDescriptor_t typeCompute_16BF = HIPTENSOR_COMPUTE_DESC_16BF;
    constexpr hiptensorComputeDescriptor_t typeCompute_16F = HIPTENSOR_COMPUTE_DESC_16F;
    constexpr hiptensorComputeDescriptor_t typeCompute_32F = HIPTENSOR_COMPUTE_DESC_32F;
    constexpr hiptensorComputeDescriptor_t typeCompute_64F = HIPTENSOR_COMPUTE_DESC_64F;

    for(int i=0; i<modesA.size(); i++)
    {
        printRunHead("Input_BF16_Compute_BF16", modesA[i], modesB[i], modesC[i], extents[i]);
        floatTypeCompute alpha{1.0f};
        floatTypeCompute beta{1.0f};
        bilinearContraction<DataType_bf16, DataType_bf16, DataType_bf16,
                            type_16BF, type_16BF, type_16BF, typeCompute_32F>
                            (&alpha, &beta, modesA[i], modesB[i], modesC[i], extents[i]);

        printRunHead("Input_F16_Compute_F16", modesA[i], modesB[i], modesC[i], extents[i]);
        floatTypeCompute alpha1{1.0f};
        floatTypeCompute beta1{1.0f};
        bilinearContraction<DataType_f16, DataType_f16, DataType_f16,
                            type_16F, type_16F, type_16F, typeCompute_32F>
                            (&alpha1, &beta1, modesA[i], modesB[i], modesC[i], extents[i]);

        printRunHead("Input_F32_Compute_BF16", modesA[i], modesB[i], modesC[i], extents[i]);
        bf16TypeCompute alpha2{1.0f};
        bf16TypeCompute beta2{1.0f};
        bilinearContraction<DataType_f32, DataType_f32, DataType_f32,
                            type_32F, type_32F, type_32F, typeCompute_16BF>
                            (&alpha2, &beta2, modesA[i], modesB[i], modesC[i], extents[i]);

        printRunHead("Input_F32_Compute_F16", modesA[i], modesB[i], modesC[i], extents[i]);
        f16TypeCompute alpha3{2.0f};
        f16TypeCompute beta3{2.0f};
        bilinearContraction<DataType_f32, DataType_f32, DataType_f32,
                            type_32F, type_32F, type_32F, typeCompute_16F>
                            (&alpha3, &beta3, modesA[i], modesB[i], modesC[i], extents[i]);

        printRunHead("Input_F32_Compute_F32", modesA[i], modesB[i], modesC[i], extents[i]);
        floatTypeCompute alpha4{1.0f};
        floatTypeCompute beta4{1.0f};
        bilinearContraction<DataType_f32, DataType_f32, DataType_f32,
                            type_32F, type_32F, type_32F, typeCompute_32F>
                            (&alpha4, &beta4, modesA[i], modesB[i], modesC[i], extents[i]);

        printRunHead("Input_F64_Compute_F32", modesA[i], modesB[i], modesC[i], extents[i]);
        floatTypeCompute alpha5{1.0f};
        floatTypeCompute beta5{1.0f};
        bilinearContraction<DataType_f64, DataType_f64, DataType_f64,
                            type_64F, type_64F, type_64F, typeCompute_32F>
                            (&alpha5, &beta5, modesA[i], modesB[i], modesC[i], extents[i]);

        printRunHead("Input_F64_Compute_F64", modesA[i], modesB[i], modesC[i], extents[i]);
        doubleTypeCompute alpha6{1.0};
        doubleTypeCompute beta6{1.0};
        bilinearContraction<DataType_f64, DataType_f64, DataType_f64,
                            type_64F, type_64F, type_64F, typeCompute_64F>
                            (&alpha6, &beta6, modesA[i], modesB[i], modesC[i], extents[i]);
    }

    return 0;
}
