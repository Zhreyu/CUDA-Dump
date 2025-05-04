/***********************************************************************
 * langtons_ant.cu  ―  CUDA Langton’s Ant with live snapshots
 *
 * Highlights
 * ----------
 * • Any number of boards/ants in parallel (one thread == one board).
 * • Print board 0 at arbitrary step counts (comma-separated list).
 * • Pinned host memory & cudaMemset for a little extra speed.
 *
 * Build
 * -----
 *   nvcc -O3 -std=c++17 langtons_ant_snapshots.cu -o langtons_ant
 *
 * CLI
 * ----
 *   ./langtons_ant [numBoards] [width] [height] [totalSteps] [snap1,snap2,…]
 *
 *   • numBoards   : default 1
 *   • width×height: default 80×40 (fits nicely in a terminal)
 *   • totalSteps  : default 10000
 *   • snapshots   : default "500,5000,10000"
 **********************************************************************/
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

/* ------------------------------------------------------------------ */
/*  Error-checking macro                                               */
/* ------------------------------------------------------------------ */
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t e = call;                                                 \
        if (e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(e));               \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

/* ------------------------------------------------------------------ */
/*  Helpers shared by host & device                                    */
/* ------------------------------------------------------------------ */
__host__ __device__ __forceinline__
int idx2D(int x, int y, int w) { return y * w + x; }

struct Ant {
    int x, y;       // position
    uint8_t dir;    // 0=N,1=E,2=S,3=W
};

/* ------------------------------------------------------------------ */
/*  Kernel: each thread owns one board + one ant                       */
/* ------------------------------------------------------------------ */
__global__
void langtons_kernel(uint8_t* boards, Ant* ants,
                     int width, int height,
                     int steps, int nBoards)
{
    for (int bid = blockIdx.x * blockDim.x + threadIdx.x;
         bid < nBoards;
         bid += blockDim.x * gridDim.x)
    {
        const size_t board_sz = static_cast<size_t>(width) * height;
        uint8_t* grid = boards + board_sz * static_cast<size_t>(bid);

        Ant ant = ants[bid];

        for (int s = 0; s < steps; ++s)
        {
            const int pos = idx2D(ant.x, ant.y, width);
            const uint8_t colour = grid[pos];

            if (colour == 0) {                 // white → right, flip black
                ant.dir = (ant.dir + 1) & 3;
                grid[pos] = 1;
            } else {                           // black → left,  flip white
                ant.dir = (ant.dir + 3) & 3;
                grid[pos] = 0;
            }
            switch (ant.dir) {                 // forward (toroidal)
                case 0:  ant.y = (ant.y - 1 + height) % height; break;
                case 1:  ant.x = (ant.x + 1) % width;           break;
                case 2:  ant.y = (ant.y + 1) % height;          break;
                case 3:  ant.x = (ant.x - 1 + width)  % width;  break;
            }
        }
        ants[bid] = ant;
    }
}

/* ------------------------------------------------------------------ */
/*  Pretty ASCII dump of a single board + its ant                      */
/* ------------------------------------------------------------------ */
void dump_board(const uint8_t* grid, const Ant& ant, int w, int h)
{
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (x == ant.x && y == ant.y) { std::putchar('A'); }
            else                           { std::putchar(grid[idx2D(x,y,w)] ? '#' : '.'); }
        }
        std::putchar('\n');
    }
}

/* ------------------------------------------------------------------ */
/*  Parse comma-separated snapshot list                                */
/* ------------------------------------------------------------------ */
std::vector<int> parse_snapshots(const char* s, int totalSteps)
{
    std::vector<int> snaps;
    std::stringstream ss(s ? s : "50000");
    std::string item;
    while (std::getline(ss, item, ',')) {
        int v = std::atoi(item.c_str());
        if (v > 0 && v <= totalSteps) snaps.push_back(v);
    }
    std::sort(snaps.begin(), snaps.end());
    snaps.erase(std::unique(snaps.begin(), snaps.end()), snaps.end());
    if (snaps.empty()) snaps = { totalSteps };
    return snaps;
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */
int main(int argc, char** argv)
{
    /* ---------------- command-line ---------------- */
    int nBoards    = (argc > 1) ? std::atoi(argv[1]) : 1;
    int width      = (argc > 2) ? std::atoi(argv[2]) : 80;
    int height     = (argc > 3) ? std::atoi(argv[3]) : 40;
    int totalSteps = (argc > 4) ? std::atoi(argv[4]) : 10000;

    const char* snapArg = (argc > 5) ? argv[5] : nullptr;
    std::vector<int> snaps = parse_snapshots(snapArg, totalSteps);

    const size_t boardCells  = static_cast<size_t>(width) * height;
    const size_t boardsBytes = boardCells * nBoards;

    /* ---------------- device allocations ---------------- */
    uint8_t* d_boards = nullptr;
    Ant*     d_ants   = nullptr;
    CUDA_CHECK(cudaMallocManaged(&d_boards, boardsBytes));       // Unified
    CUDA_CHECK(cudaMallocManaged(&d_ants, sizeof(Ant) * nBoards));

    /* initialise boards & ants */
    CUDA_CHECK(cudaMemset(d_boards, 0, boardsBytes));
    for (int b = 0; b < nBoards; ++b)
        d_ants[b] = { width / 2, height / 2, 0 };

    /* host-side pinned buffer for board 0 */
    uint8_t* h_board0 = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_board0, boardCells, cudaHostAllocDefault));

    /* ---------------- launch config ---------------- */
    const int T = 256;
    const int B = (nBoards + T - 1) / T;

    int prev = 0;
    for (int snap : snaps)
    {
        int chunk = snap - prev;
        prev = snap;

        langtons_kernel<<<B, T>>>(d_boards, d_ants,
                                  width, height,
                                  chunk, nBoards);
        CUDA_CHECK(cudaDeviceSynchronize());

        /* copy board 0 + its ant back to host and print */
        std::memcpy(h_board0, d_boards, boardCells);
        std::printf("\n===== Snapshot after %d steps =====\n", snap);
        dump_board(h_board0, d_ants[0], width, height);
    }

    /* ---------------- cleanup ---------------- */
    CUDA_CHECK(cudaFree(d_boards));
    CUDA_CHECK(cudaFree(d_ants));
    CUDA_CHECK(cudaFreeHost(h_board0));
    return 0;
}
