#include <cstdlib>
#include <cstdio>
#include <fstream>

using namespace std;

__managed__ int nx = 41;
__managed__ int ny=41;
__managed__ int nt = 500;
__managed__ int nit = 50;
__managed__ double dx;
__managed__ double dy;
__managed__ double dt = .01;
__managed__ double rho = 1.;
__managed__ double nu = .02;

__global__ void u_v_p_b_init(float *u, float *v, float *p, float *b){
    u[threadIdx.x] = 0;
    v[threadIdx.x] = 0;
    p[threadIdx.x] = 0;
    b[threadIdx.x] = 0;
}

__global__ void b_calc(float *b, float *u, float *v){
    if(threadIdx.x/ny == 0 || threadIdx.x%ny == 0 || threadIdx.x%ny == ny-1 || threadIdx.x/ny == nx-1) return ;
    int x = threadIdx.x;
    b[x] = rho * (1/dt*((b[x+1] - b[x-1])/(2*dx) + (v[x+ny] - v[x-ny])/(2*dy)) - ((u[x+1] - u[x-1])/(2*dx))*((u[x+1] - u[x-1])/(2*dx)) - 2*((u[x+ny] - u[x-ny])/(2*dy)*(v[x+1] - v[x-1])/(2*dx)) - ((v[x+ny] - v[x-ny])/(2*dy))*((v[x+ny] - v[x-ny])/(2*dy)));
}

__global__ void pn_p_copy(float *pn, float *p){
    pn[threadIdx.x] = p[threadIdx.x];
}

__global__ void p_calc(float *p, float *pn, float *b){
    if(threadIdx.x/ny == 0 || threadIdx.x%ny == 0 || threadIdx.x%ny == ny-1 || threadIdx.x/ny == nx-1) return ;
    int x = threadIdx.x;
    p[x] = dy*dy*(pn[x+1] + pn[x-1]) + dx*dx*(pn[x+ny] + pn[x-ny]) - b[x]*dx*dx*dy*dy/(2*dx*dx + 2*dy+dy);
}

__global__ void p_op_1(float *p){
    p[threadIdx.x + nx - 1] = p[threadIdx.x + nx -2];
    p[threadIdx.x] = p[threadIdx.x + 1];
}

__global__ void p_op_2(float *p){
    p[(ny-1)*ny + threadIdx.x] = 0;
    p[threadIdx.x] = p[ny + threadIdx.x];
}

__global__ void un_u_vn_v_copy(float *un, float *u, float *vn, float *v){
    un[threadIdx.x] = u[threadIdx.x];
    vn[threadIdx.x] = v[threadIdx.x];
}

__global__ void u_v_calc(float *u, float *v, float *un, float *vn, float *p){
    if(threadIdx.x/ny == 0 || threadIdx.x%ny == 0 || threadIdx.x%ny == ny-1 || threadIdx.x/ny == nx-1) return ;
    int x = threadIdx.x;
    u[x] = un[x] - un[x]*dt/dx*(un[x] - un[x-1]) - un[x]*dt/dy*(un[x] - un[x-ny]) - dt/(2*rho*dx)*(p[x+1] - p[x-1])
            + nu*dt/(dx*dx)*(un[x+1] - 2*un[x] + un[x-1]) + nu*dt/(dy*dy)*(un[x+ny] - 2*un[x] + un[x-ny]);
    v[x] = vn[x] - vn[x]*dt/dx*(vn[x] - vn[x-1]) - vn[x]*dt/dy*(vn[x] - vn[x-ny]) - dt/(2*rho*dx)*(p[x+1] - p[x-1])
            + nu*dt/(dx*dx)*(vn[x+1] - 2*vn[x] + vn[x-1]) + nu*dt/(dy*dy)*(vn[x+ny] - 2*vn[x] + vn[x-ny]);
}

__global__ void u_v_op_1(float *u, float *v){
    u[threadIdx.x*ny] = 0;
    v[threadIdx.x*ny] = 0;
    u[threadIdx.x*ny + nx-1] = 0;
    v[threadIdx.x*ny + nx-1] = 0;
}

__global__ void u_v_op_2(float *u, float *v){
    u[threadIdx.x] = 0;
    u[(ny-1)*nx + threadIdx.x] = 1;
    v[threadIdx.x] = 0;
    v[(ny-1)*nx + threadIdx.x] = 0;
    
}

int main() {
//   nx = 41;
//   ny = 41;
//   nt = 500;
//   nit = 50;
  dx = 2. / (nx - 1);
  dy = 2. / (ny - 1);
//   dt = .01;
//   rho = 1.;
//   nu = .02;

  float *u, *v, *p, *b, *un, *vn, *pn;
  cudaMallocManaged(&u, ny*nx*sizeof(int));
  cudaMallocManaged(&v, ny*nx*sizeof(int));
  cudaMallocManaged(&p, ny*nx*sizeof(int));
  cudaMallocManaged(&b, ny*nx*sizeof(int));
  cudaMallocManaged(&un, ny*nx*sizeof(int));
  cudaMallocManaged(&vn, ny*nx*sizeof(int));
  cudaMallocManaged(&pn, ny*nx*sizeof(int));

  u_v_p_b_init<<<1, nx*ny>>>(u, v, p, b);

  ofstream ufile("u_cu.dat");
  ofstream vfile("v_cu.dat");
  ofstream pfile("p_cu.dat");

  for(int n=0; n<nt; n++){
    b_calc<<<1, nx*ny>>>(b, u, v);

    for(int it=0; it<nit; it++){
        pn_p_copy<<<1, nx*ny>>>(p, pn);
        p_calc<<<1, nx*ny>>>(p, pn, b);
        p_op_1<<<1, ny>>>(p);
        p_op_2<<<1, nx>>>(p);
    }

    un_u_vn_v_copy<<<1, nx*ny>>>(un, u, vn, v);

    u_v_calc<<<1, nx*ny>>>(u, v, un, vn, p);

    u_v_op_1<<<1, ny>>>(u, v);

    u_v_op_2<<<1, nx>>>(u, v);

    if (n % 10 == 0) {
      for (int x=0; x<nx*ny; x++)  ufile << u[x] << " ";
      ufile << "\n";
      for (int x=0; x<nx*ny; x++)  vfile << v[x] << " ";
      vfile << "\n";
      for (int x=0; x<nx*ny; x++)  pfile << p[x] << " ";
      pfile << "\n";
    }
  }

  ufile.close();
  vfile.close();
  pfile.close();
}