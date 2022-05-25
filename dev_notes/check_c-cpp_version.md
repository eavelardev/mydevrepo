# dev-tools
Tools for dev stuff

## Check C/C++ version

During build time

```C
#if (__STDC_VERSION__ == 201710L)
	#pragma message ( "-std=c17" )
#elif (__STDC_VERSION__ == 201112L)
	#pragma message ( "-std=c11" )
#elif (__STDC_VERSION__ == 199901L)
	#pragma message ( "-std=c99" )
#endif

#if (__cplusplus == 201703L)
	#pragma message ("-std=c++17" )
#elif (__cplusplus == 201402L)
	#pragma message ( "-std=c++14" )
#elif (__cplusplus == 201103L)
	#pragma message ( "-std=c++11" )
#elif (__cplusplus == 199711L)
	#pragma message ( "-std=c++98" )
#endif
```

During execution time

```C
if (__STDC_VERSION__ == 201710L) printf("-std=c17\n");
else if (__STDC_VERSION__ == 201112L) printf("-std=c11\n");
else if (__STDC_VERSION__ == 199901L) printf("-std=c99\n");

if (__cplusplus == 201703L) std::cout << "-std=c++17\n";
else if (__cplusplus == 201402L) std::cout << "-std=c++14\n";
else if (__cplusplus == 201103L) std::cout << "-std=c++11\n";
else if (__cplusplus == 199711L) std::cout << "-std=c++98\n";
```
