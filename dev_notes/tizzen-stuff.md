# tizen-stuff

## Tizen Native C

Tizen v5.0
Default language standard: -std=c11

Using Complex numbers
```c
#include "tgmath.h"

double complex exponential = exp(I*2*M_PI*(f0 + k/2*t)*t);
```

## Tizen Native C++

Tizen v5.0
Default language standard: -std=c++14

Using Complex numbers, `#include <ctgmath>`

Toolchain: GCC-6.2
```c++
complex<double> exponential1 = exp<double>(1i*2*M_PI*(f0 + k/2*t)*t);
complex<double> exponential2 = exp<double>(1i*2.0*M_PI*(f0 + k/2*t)*t);
```

Toolchain: LLVM-4.0 with GCC-6.2
```c++
complex<double> exponential1 = exp(1i*2.0*M_PI*(f0 + k/2*t)*t);
complex<double> exponential2 = exp<double>(1i*2.0*M_PI*(f0 + k/2*t)*t);
```
