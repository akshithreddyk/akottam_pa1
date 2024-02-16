#include <iostream>
#include <cuda.h>
#include <random>
#include <fstream>
#include <iomanip>


// The cuda kernel
__global__ void quamsim_kernel(const float *input_state, float *output_state, const float *gate_matrix, int n, int t) {
  
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int target_state = 1 << t;
  int update_i1 = i;
  int update_i2 = i ^ target_state;

  //if ( (i <= (n - 1)) &&  (((i/target_state)%2) != 1) ) {
  if ( (i <= (n - 1)) &&  (!(i & target_state))) {
    output_state[update_i1] = gate_matrix[0] * input_state[update_i1] + gate_matrix[1] * input_state[update_i2];
    output_state[update_i2] = gate_matrix[2] * input_state[update_i1] + gate_matrix[3] * input_state[update_i2];
  }

}

int main(int argc, char *argv[]) {

  cudaError_t err = cudaSuccess;
  // Read the inputs from command line
  char *filename;
  //std::string filename = argv[1];
  filename = argv[1];

  std::ifstream input_file(filename);

  //Reads the gate matrix
  float gate_matrix[4];
  for (int i=0; i<4; i++) {
    input_file >> gate_matrix[i];
  }

  //Reads the quantum state
  std::vector<float> state;
  float state_val;
  while (input_file >> state_val) {
    state.push_back(state_val);
  }

  //Read the target qubit
  int target_q = state.back();
  state.pop_back();

  int n = state.size();
  int N = log2(n);

  size_t size_s = n * sizeof(float);
  size_t size_g = 4 * sizeof(float);

  // Allocate/move data using cudaMalloc and cudaMemCpy
  //Allocate the host memory

  float *h_gate_matrix = (float *)malloc(4 * sizeof(float));
  float *h_input_state = (float *)malloc(n * sizeof(float));
  float *h_output_state = (float *)malloc(n * sizeof(float));
  // Verify that allocations succeeded
  if (h_gate_matrix == NULL || h_input_state == NULL || h_output_state == NULL) {
      fprintf(stderr, "Failed to allocate host vectors!\n");
      exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors
  for (int i = 0; i < 4; ++i) {
      h_gate_matrix[i] = gate_matrix[i];
  }
  for (int i = 0; i < n; ++i) {
      h_input_state[i] = state[i];
  } 

  // Allocate the device
  float *d_gate_matrix = NULL;
  err = cudaMalloc((void **)&d_gate_matrix, size_g);
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  float *d_input_state = NULL;
  err = cudaMalloc((void **)&d_input_state, size_s);
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  float *d_output_state = NULL;
  err = cudaMalloc((void **)&d_output_state, size_s);
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  cudaMemcpy(d_gate_matrix, h_gate_matrix, size_g, cudaMemcpyHostToDevice);
  cudaMemcpy(d_input_state, h_input_state, size_s, cudaMemcpyHostToDevice);

  // Launch the kernel
  int threadsPerBlock = 256;
  int blocksPerGrid =(n + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
  quamsim_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input_state, d_output_state, d_gate_matrix, n, target_q);
    
  cudaMemcpy(h_output_state, d_output_state, size_s, cudaMemcpyDeviceToHost);

  // Print the output
  for (int i = 0; i < n; ++i) {
    std::cout<<h_output_state[i]<<std::endl;
  }

  // Clean up the memory
  cudaFree(d_gate_matrix);
  cudaFree(d_input_state);
  cudaFree(d_output_state);
  free(h_gate_matrix);
  free(h_input_state);
  free(h_output_state);

  cudaDeviceReset();

  return 0;
}
