#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "sw_allall_flip_conditionals.h"

int ind(int row, int col, int nn)
{
    return col*nn + row;
}

int main()
{
    int i;
    int nn = 4;
    double* biases =  malloc(nn*sizeof(double));
    double* oldbiases =  malloc(nn*sizeof(double));
    double* bonds = calloc(nn*nn, sizeof(double));
    int* ptr = malloc(nn*sizeof(int));
    double* flips = malloc(nn*sizeof(double));
    double* cprobs = malloc(nn*sizeof(double));

    /* testcase 1 */
    biases[0] = -2.;
    biases[1] = -3.;
    biases[2] = -5.;
    biases[3] = -7.;
    for (i=0; i < nn; ++i)
        oldbiases[i] = biases[i];
    bonds[ind(0, 3, nn)] = 1;
    bonds[ind(1, 2, nn)] = 1;
    percolate(bonds, ptr, biases, nn);
    flip_clusters(flips, cprobs, ptr, biases, nn);
    for (i=0; i < nn; ++i) {
        printf("i=%d, flips[%d]=%g, ptr[%d]=%d, biases[%d]=%g\n", i, i, flips[i], i, ptr[i], i, biases[i]);
    }
    printf("\n");
    assert(ptr[0]==-2); assert(biases[0]==oldbiases[0]+oldbiases[3]);
    assert(ptr[1]==-2); assert(biases[1]==oldbiases[1]+oldbiases[2]);
    assert(ptr[2]== 1);
    assert(ptr[3]== 0);

    /* testcase 2 */
    biases[0] = 2.;
    biases[1] = 3.;
    biases[2] = 5.;
    biases[3] = 7.;
    bonds[ind(0, 3, nn)] = 0;
    bonds[ind(1, 2, nn)] = 0;
    bonds[ind(0, 3, nn)] = 1;
    bonds[ind(1, 3, nn)] = 1;
    percolate(bonds, ptr, biases, nn);
    flip_clusters(flips, cprobs, ptr, biases, nn);
    for (i=0; i < nn; ++i) {
        printf("i=%d, flips[%d]=%g, ptr[%d]=%d, biases[%d]=%g\n", i, i, flips[i], i, ptr[i], i, biases[i]);
    }
    printf("\n");
    assert(ptr[0]==-3); assert(biases[0]==12.);
    assert(ptr[2]==-1); assert(biases[2]==5.);
    assert(ptr[1]== 0);
    assert(ptr[3]== 0);

    free(biases);
    free(oldbiases);
    free(bonds);
    free(flips);
    free(cprobs);
    free(ptr);
    return 0;
}

