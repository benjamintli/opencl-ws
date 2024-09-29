#define TS 16

__kernel void matmul(
    const int m,
    const int n,
    const int k,
    const __global float* a,                      
    const __global float* b,                      
    __global float* c)            
{
    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = TS * get_group_id(0) + row;
    const int globalCol = TS * get_group_id(1) + col;

    __local float a_sub[TS][TS];
    __local float b_sub[TS][TS];

    float acc = 0.0f;
    const int numTiles = k / TS;
    for (int t = 0; t < numTiles; t++) {
        const int tiledRow = TS * t + row;
        const int tiledCol = TS * t + col;
        a_sub[col][row] = a[tiledCol * m + globalRow];
        b_sub[col][row] = b[globalCol * k + tiledRow];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int _k = 0; _k < TS; _k++) {
            acc += a_sub[_k][row] * b_sub[col][_k];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    c[globalCol * m + globalRow] = acc;
}