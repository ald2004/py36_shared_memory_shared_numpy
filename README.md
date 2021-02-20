
##### from  process a
```python
import shared_numpy as snp
import numpy as np
input_shape=(2,3,4)
a = snp.array(np.random.randn(*input_shape) ,dtype=np.float32)
>>> a.shm.name
'psm_103548f6'

a.close()
a.unlink()
del a
```

##### from process b
```python
shm_b = shared_memory.SharedMemory('psm_103548f6')
shm_arr = snp.SharedNDArray(input_shape,dtype='float32',buffer=shm_b.buf)

shm_arr.close()
shm_arr.unlink()
del shm_arr
del shm_b
```
##### from list
```python
import shared_memory
a = shared_memory.ShareableList(['howdy', b'HoWdY', -273.154, 100, None, True, 42])
b = shared_memory.ShareableList(range(5))         # In a first process
c = shared_memory.ShareableList(name=b.shm.name)  # In a second process
```
