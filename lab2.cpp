#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <math.h>

//generate some random numbers 
#define RAND_MAX_GEN 50.0

double random_double();
double* generate_random(size_t num, int rank);
double find_sum(double* numbers, size_t size);

void start_find_sum(int rank, int size,
                    double* random_numbers,
                    size_t num_per_proc,
                    double* overall_sum);

void start_find_psum(int rank, int size,
     double* random_numbers, size_t num_per_proc,
     double sum);

int main(int argc, char **argv)
{
  int rank, size;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Barrier(MPI_COMM_WORLD);

  /* `num` is the total number of random numbers between all
   * processes; allow an optional parameter to set `num`
   */
  size_t num = 90000000;
  size_t pos = 5;
  if (argc == 3) { 
    num = (size_t) atoi(argv[1]);
    pos = (size_t) atoi(argv[2]);
  }

  /* generate some random numbers for `rank` */
  size_t num_per_proc = num / size;
  double* random_numbers = generate_random(num_per_proc,
                                           rank);

  #ifdef DEBUG
    if (rank == 0)
      printf("Generated %zd random numbers in each process\n",
             num_per_proc);
  #endif

  double sum;
  double elapsed = 0;

  double start = MPI_Wtime();
  start_find_sum(rank, size, random_numbers, num_per_proc,
                 &sum);
  start_find_psum(rank, size, random_numbers, num_per_proc,
                  sum);

  
  MPI_Barrier(MPI_COMM_WORLD);

  
  size_t start_pos = num_per_proc * rank;
  size_t end_pos = num_per_proc * (rank+1) - 1;
  if (pos >= start_pos && pos <= end_pos) {
    printf("Prefix sum number %zu is %lf\n",
          pos, random_numbers[pos - start_pos]);
  }
  double end = MPI_Wtime();

  if (rank == 0) {
    elapsed = end - start;
    printf("Took %lf seconds\n", elapsed);
  }

  #ifdef DEBUG
    size_t j;
    for (j = 0; j < num_per_proc; j++) {
      size_t j_real_pos = start_pos + j;
      printf("Prefix sum %02zu: %lf\n", j_real_pos,
            random_numbers[j]);
    }
  #endif

  free(random_numbers);
  MPI_Finalize();
  return 0;
}

// return a random double in [0.0, 1.0)
double random_double()
{
  return (double)random() /
          ((double)RAND_MAX + (double)1);
}


double* generate_random(size_t num, int rank)
{
  
  srandom(time(NULL) + rank);

  double* random_numbers = (double*)
                          malloc(sizeof(double) * num);
  if (!random_numbers) {
    printf("%d Failed to malloc\n", rank);
    exit(EXIT_FAILURE);
  }

  for (size_t i = 0; i < num; i++) {
    random_numbers[i] = random_double() * RAND_MAX_GEN; 
                       
    #ifdef DEBUG
      size_t i_real_pos = num * rank + i;
      printf("Number %02zu: %lf\n", i_real_pos,
            random_numbers[i]);
    #endif

  }
  return random_numbers;
}



double find_sum(double* numbers, size_t size)
{
  double sum = 0.0;
  for (size_t i = 0; i < size; i++) {
    sum += numbers[i];
  }
  return sum;
}


void start_find_sum(int rank, int size,
     double* random_numbers, size_t num_per_proc,
     double* overall_sum)
{
  MPI_Status status;

  double sum = find_sum(random_numbers, num_per_proc);

  int still_alive = 1;
  int level;

  

  for (level = 0; level < (int)log2(size); level++) {
    if (still_alive) {
      int position = rank / (int)pow(2, level);

      if (position % 2 == 0) {
        //  receiver
        double sender_sum;
        int sending_rank = rank + (int)pow(2, level);

        MPI_Recv(&sender_sum, 1, MPI_DOUBLE, sending_rank,
                 0, MPI_COMM_WORLD, &status);

        sum += sender_sum;
      }

      else {
        //  sender
        int receiving_rank = rank - (int)pow(2, level);

        MPI_Send(&sum, 1, MPI_DOUBLE, receiving_rank, 0,
                 MPI_COMM_WORLD);
        still_alive = 0;
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  *overall_sum = sum;
}

/*
 * Description of start_find_psum
 * A process with rank R on level i that is an even-numbered
 * position as its sum set to the sum of this process R on
 * level i + 1, rather than level i. This sum was created by
 * adding the old sum from level i with the sum of process
 * R + 2^i (the sibling of process R). The correct sum for
 * process R on level i can be restored by subtracting away
 * the sum of the sibling process of R.
 *
 * (The above about restoring the correct sum isn't necessary and is
 * commented out.)
 *
 * Each node (process) in the tree has an associated prefix
 * sum that represents the prefix sum of all the numbers up
 * to the last number of the rightmost leaf process in the
 * tree rooted at that node. Rank 0 is the root node and so
 * its psum is set to sum.
 *
 * Iteration is from the level under the root to the bottom
 * level. A node in an odd-numbered position on the current
 * level receives its prefix sum from its parent.  Because
 * even-numbered nodes become parents on the next level, the
 * parent of any odd numbered node of rank R on level i is R
 * - 2^i. A node in an even-numbered position on level i has
 * its prefix sum set as the prefix sum of its parent minus
 * the regular sum of its sibling.  Because a node in an
 * even-numbered position is also its own parent, it already
 * has the prefix sum of its parent. It also has access to
 * the regular sum of its sibling from the time it fixed its
 * own sum, so that value can be reused.
 *
 * Now every node of rank R has its prefix sum; this
 * prefix sum is the sum of all of the `random_numbers` of
 * nodes from Rank 0 to Rank R. We will overwrite
 * `random_numbers` to be the prefix sums for
 * num_per_proc*rank  TO   num_per_proc*(rank + 1) - 1,
 * from a global perspective, if we consider each
 * random_numbers as part of a distributed array.
 */
void start_find_psum(int rank, int size,
     double* random_numbers, size_t num_per_proc,
     double sum)
{
  double psum;
  int level;
  MPI_Status status;

  if (rank == 0) {
    psum = sum;
  }
  for (level = (int)log2(size) - 1; level >= 0; level--) {

    // triggers the process on same level
    if (level == 0 || rank % (int)pow(2, level) == 0) {
      int position = rank / (int)pow(2, level);

      if (position % 2 == 0) {
        double sender_sum;
        int sending_rank = rank + (int)pow(2, level);

      
        MPI_Send(&psum, 1, MPI_DOUBLE,
                 sending_rank, 
                 0, MPI_COMM_WORLD);

        MPI_Recv(&sender_sum, 1, MPI_DOUBLE,
                 sending_rank, 0, MPI_COMM_WORLD, &status);

        // fixes this sum to be the correct value

        psum -= sender_sum;
      }

      else{
        int receiving_rank = rank - (int)pow(2, level);

        MPI_Recv(&psum, 1, MPI_DOUBLE,
                 receiving_rank, // PARENT
                 0, MPI_COMM_WORLD, &status);

        // send sum to receiving_rank so it can fix its sum
        MPI_Send(&sum, 1, MPI_DOUBLE,
                 receiving_rank, 0, MPI_COMM_WORLD);
      }

    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  // places the prefix  of sums associated with this node in all random_numbers
  double next_sum = random_numbers[num_per_proc-1];
  random_numbers[num_per_proc-1] = psum;

  int j;
  for (j = num_per_proc - 2; j >= 0; j--) {
    double next_sum_tmp = random_numbers[j];

    random_numbers[j] = random_numbers[j+1] -
                        next_sum;

    next_sum = next_sum_tmp;
  }
}