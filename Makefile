all:
	x86_64-linux-gnu-gcc -pthread -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC -DHAVE_SHM_OPEN=1 -DHAVE_SHM_UNLINK=1 -DHAVE_SHM_MMAN_H=1 -I/usr/include/python3.6m -c posixshmem.c -o posixshmem.o
	x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -Wl,-Bsymbolic-functions -Wl,-z,relro -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 posixshmem.o -lrt -o _posixshmem.cpython-36m-x86_64-linux-gnu.so

clean:
	rm -rf ./_posixshmem.cpython-36m-x86_64-linux-gnu.so ./posixshmem.o ./__pycache__
