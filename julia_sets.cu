// Math                 --> GPU Mapping
// Complex gird (image) --> thred-block grid
// Constant C         --> __constant__ memory
// z = z^2 + c > MAX_I --> Inner while loop each thred 
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
// Define constants
#define DIM 1000 // Image dimensions



// Complex number structure
struct Complex {
    float r;  // real part
    float i;  // imaginary part
};

void writePPM(const char *filename, unsigned char *img) {
    FILE *fp = fopen(filename, "wb");
    if(!fp) { perror("fopen"); exit(1); }
    // P6 = binary RGB
    fprintf(fp, "P6\n%d %d\n255\n", DIM, DIM);
    for(int y = 0; y < DIM; y++) {
        for(int x = 0; x < DIM; x++) {
            int idx = 4*(x + y*DIM);
            unsigned char rgb[3] = { img[idx+0], img[idx+1], img[idx+2] };
            fwrite(rgb, 1, 3, fp);
        }
    }
    fclose(fp);
}

struct CPUBitmap {
    int   w, h;
    unsigned char *ptr;
    CPUBitmap(int W,int H): w(W), h(H) {
        ptr = (unsigned char*)malloc(w*h*4);
    }
    ~CPUBitmap(){ free(ptr); }
    unsigned char *get_ptr() { return ptr; }
    size_t image_size() const { return w*h*4; }
    void display_and_wait() {
        writePPM("out.ppm", ptr);
        printf("Wrote out.ppm\n");
        system("eog out.ppm"); //to auto‑open

    }
};


// Helper functions for complex arithmetic
__device__ void add(Complex a, Complex b, Complex *c) {
    c->r = a.r + b.r;
    c->i = a.i + b.i;
}

__device__ void mul(Complex a, Complex b, Complex *c) {
    c->r = a.r * b.r - a.i * b.i;
    c->i = a.r * b.i + a.i * b.r;
}

__device__ float magnitude(Complex a) {
    return a.r * a.r + a.i * a.i;
}

__device__ int julia(int x, int y){ //CPU function
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x) / (float)(DIM/2);
    float jy = scale * (float)(DIM/2 - y) / (float)(DIM/2);

    struct Complex c, a, r1, r2;
    c.r = -0.07; c.i = 0.69;
    a.r = jx; a.i = jy;
    int i = 0;

    for(i = 0; i < 500; i++){
        mul(a,a,&r1);
        add(r1,c,&r2);
        if (magnitude(r2) > 1000) 
            return 0;
        a.r = r2.r; a.i = r2.i;
    }
    return 1;
}


__global__ void kernal (unsigned char *ptr){
    // map from threadIdx/ BloxkIdx to pixel position

    int  x = blockIdx.x;
    int  y = blockIdx.y;
    int offset = x + y * DIM;

    int juliaValue = julia(x,y);
    ptr[offset * 4 + 0] = 255 * juliaValue; // red , black if 0
    ptr[offset * 4 + 1] = 121; // green
    ptr[offset * 4 + 2] = 125; // blue
    ptr[offset * 4 + 3] = 125; // transparency
}

// void kernal(unsigned char *ptr){ //cpu function
//     for(int y=0; y < DIM; y++){
//         for(int x=0; x < DIM; x++){
//            int offset = x + y * DIM;
//            int juliaValue = julia(x,y);
           // each pixel has 4 channels (RGBA) therefore we need to multiply offset by 4 //
           // ptr array is representing the bitmap for image //
//            ptr[offset * 4 + 0] = 255 * juliaValue; // red
//            ptr[offset * 4 + 1] = 0; // green
//            ptr[offset * 4 + 2] = 0; // blue
//            ptr[offset * 4 + 3] = 255; // transparency
//         }
//     }
// }




// int main(void){
//     // Uncomment and include the cpu_bitmap.h header when needed
//     CPUBitmap bitmap(DIM, DIM);
//     unsigned char *ptr = bitmap.get_ptr();
//     kernal(ptr);
//     bitmap.display_and_wait();
//     return 0;
// }

int main(void) {
    // Allocate memory for the image
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;
    cudaMalloc((void**)&dev_bitmap, bitmap.image_size());


    // Define grid and block dimensions
    dim3 grid(DIM, DIM);

    kernal<<<grid, 1>>>(dev_bitmap);

    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
    bitmap.display_and_wait();
    cudaFree(dev_bitmap);

    return 0;
}