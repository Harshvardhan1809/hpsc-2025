#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {

    // 座標のベクトルを作成する
    __m512 x_vec = _mm512_load_ps(&x);
    __m512 y_vec = _mm512_load_ps(&y);

    // 実質的にjが表している要素のベクトル
    __m512 x_plus = _mm512_set1_ps(x[i]);
    __m512 y_plus = _mm512_set1_ps(y[i]);
    __m512 m_vec = _mm512_set1_ps(m[i]);

    // マスクベクトルの生成
    float mask_arr[N] = {1};
    mask_arr[i] = 0;
    __m512 mask = _mm512_load_ps(&mask_arr);

    __m512 rxvec = _mm512_sub_ps(x_vec, _mm512_mul_ps(x_plus, mask));
    __m512 ryvec = _mm512_sub_ps(y_vec, _mm512_mul_ps(y_plus, mask));

    __m512 r2vec = _mm512_add_ps(_mm512_mul_ps(rxvec, rxvec), _mm512_mul_ps(ryvec, ryvec));
    __m512 r3vec = _mm512_mul_ps(_mm512_rsqrt14_ps(r2vec), r2vec);

    // 力の計算
    __m512 fxvec = _mm512_div_ps(_mm512_mul_ps(rxvec, m_vec), r3vec);
    __m512 fyvec = _mm512_div_ps(_mm512_mul_ps(ryvec, m_vec), r3vec);

    // 計算結果をreduceして元の配列に保存する
    fx[i] = _mm512_reduce_add_ps(fxvec);
    fy[i] = _mm512_reduce_add_ps(fyvec);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
