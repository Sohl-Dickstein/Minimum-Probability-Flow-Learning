#include <stdlib.h>
#include <math.h>

/* Swendsen-Wang helper code for binary models with pairwise correlations (Known
 * as Ising models, Boltzmann machines, binary potts models, etc.)
 *
 * See the calling Matlab code for the bigger picture of the S-W algorithm. This
 * C code works out the clusters given the bonds provided by the Matlab code. It
 * then works out the probability of flipping each cluster given biases passed
 * by Matlab, makes a decision for each node and passes out both the decision
 * and the conditional probability. The Matlab code can then turn this flip
 * information into actual settings of the Ising model.
 *
 * Here the graph of variables is assumed to be all-all connected. And the
 * implementation is fairly naive. If many of the connections are weak, it would
 * be possible to reduce the computational complexity of each update
 * dramatically with a little pre-processing. Note that many of the connections
 * better be weak, or percolation will be above a phase transition, the system
 * will all lock together and the sampler will do nothing interesting!
 *
 * bonds and flips are stored as doubles throughout because Matlab makes
 * dealing with non-doubles difficult and I couldn't be bothered with
 * conversions
 */


int
findroot(int *ptr, int i)
/* Adapted from:
 *     A fast Monte Carlo algorithm for site or bond percolation
 *     M. E. J. Newman and R. M. Ziff
 *     arXiv:cond-mat/0101295 v2 8th April 2001
 *
 * See that for an explanation of ptr[]
 */
{
    if (ptr[i]<0)
        return i;
    return ptr[i] = findroot(ptr, ptr[i]);
}


void
percolate(double* bonds, int *ptr, double* biases, int nn)
/* This routine populates ptr[], from which the clustering implied by the bonds
 * can be colored more easily. The biases of the nodes are destructively accumulated:
 * the root of each cluster has the total bias in biases, any other entries are
 * arbitrary values.
 */
{
    int i, j;
    int base_idx;
    int l, m;

    for (i=0; i<nn; ++i)
        ptr[i] = -1; /* All points start as root nodes in a cluster of size one */

    /* Loop over the top diagonal part of bonds matrix, percolating whenever
     * there is a bond */
    for (i=1; i < nn; ++i) {
        base_idx = i*nn;
        for (j=0; j < i; ++j) {
            if (!(bonds[base_idx+j]))
                continue;
            l = findroot(ptr, i);
            m = findroot(ptr, j);
            if (l != m) {
                /* merge separate clusters */
                if (ptr[l] < ptr[m]) {
                    /* cluster l is bigger than cluster m */
                    biases[l] += biases[m];
                    ptr[l] += ptr[m];
                    ptr[m] = l;
                } else {
                    /* cluster m is bigger than cluster l */
                    biases[m] += biases[l];
                    ptr[m] += ptr[l];
                    ptr[l] = m;
                }
            }
        }
    }
}


void
flip_clusters(double* flips, double* cprobs, int* ptr, double* biases, int nn)
/* 
 * Once ptr[] has been populated by percolate() this function decides which
 * clusters to flip (and reports the probability of flipping).
 * All nodes within a cluster are flipped (or not) together. The clusters are
 * independent with flip probability proportional to exp(total_bias).
 */
{
    int i, root;

    for (i=0; i < nn; ++i)
        flips[i] = -1;

    for (i=0; i < nn; ++i) {
        root=findroot(ptr, i);
        if (flips[root] == -1) {
            cprobs[root] = 1.0/(1.0+exp(-biases[root]));
            flips[root] = (((double)rand())/RAND_MAX) < cprobs[root];
        }
        cprobs[i] = cprobs[root];
        flips[i] = flips[root];
    }
}


/*******************************************
 * MATLAB STUFF            
 *******************************************/
#ifndef MYTESTING

#include "mex.h"

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray*prhs[] )
{
    int nn;
    int is_correct_col_vector;
    int is_correct_row_vector;
    double* biases;
    double* bonds;
    double* flips;
    double* cprobs;
    int* ptr;
    
    /*** Argument parsing and checking ***/
    if (nrhs != 2)
        mexErrMsgTxt("Usage: [flips, cprobs] = sw_allall_flip_conditionals(bonds, biases);\n");
    if (nlhs != 2)
        mexErrMsgTxt("Wrong number of output arguments.\n");
    /*
    * TODO full argument checking
    */
    
    nn = mxGetN(prhs[0]);
    if (nn != mxGetM(prhs[0]))
        mexErrMsgTxt("Bonds matrix should be square.\n");
    is_correct_row_vector = (mxGetM(prhs[1]) == nn) && (mxGetN(prhs[1]) == 1);
    is_correct_col_vector = (mxGetM(prhs[1]) == 1) && (mxGetN(prhs[1]) == nn);
    if (! (is_correct_col_vector || is_correct_row_vector))
        mexErrMsgTxt("Biases should be a vector with length matching bonds.\n");
    bonds = mxGetPr(prhs[0]);
    biases = mxGetPr(prhs[1]);

    /* Create storage for answer */
    plhs[0] = mxCreateDoubleMatrix(nn, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(nn, 1, mxREAL);
    flips = mxGetPr(plhs[0]);
    cprobs = mxGetPr(plhs[1]);
    
    /*** Get answer ***/
    ptr = malloc(nn*sizeof(int));
    percolate(bonds, ptr, biases, nn);
    flip_clusters(flips, cprobs, ptr, biases, nn);
    free(ptr);
}

#endif

