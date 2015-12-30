#include <ctime>
#include <random>
#include <cstdlib>


///////////////////////////////////////////////////////////////////////
// Timer
std::clock_t tic_toc_timer;
void tic() {
  tic_toc_timer = clock();
}
void toc() {
  std::clock_t toc_timer = clock();
  printf("Elapsed time is %f seconds.\n", double(toc_timer - tic_toc_timer) / CLOCKS_PER_SEC);
}

///////////////////////////////////////////////////////////////////////
// Run a system command
void sys_command(std::string str) {
  if (system(str.c_str()))
    return;
}

///////////////////////////////////////////////////////////////////////
// Generate random float
float gen_random_float(float min, float max) {
    
  float tmp = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  float rand_val = tmp*(max-min-0.0001) + min;

  // std::random_device rd;
  // std::mt19937 mt(rd());
  // std::uniform_real_distribution<double> dist(min, max - 0.0001);
  // float rand_val = (float) dist(mt);
  // std::cout << rand_val << std::endl;

  return rand_val;


}

// ///////////////////////////////////////////////////////////////////////
// // Generate random string
// std::string gen_rand_str(size_t len) {
//   auto randchar = []() -> char {
//     // const char charset[] =
//     //   "0123456789"
//     //   "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
//     //   "abcdefghijklmnopqrstuvwxyz";
//     const char charset[] =
//       "0123456789"
//       "abcdefghijklmnopqrstuvwxyz";
//     // const size_t max_index = (sizeof(charset) - 1);
//     // return charset[rand() % max_index];
//     return charset[((int) std::floor(gen_random_float(0.0f, (float) sizeof(charset) - 1)))];
//   };
//   std::string str(len, 0);
//   std::generate_n(str.begin(), len, randchar);
//   return str;
// }