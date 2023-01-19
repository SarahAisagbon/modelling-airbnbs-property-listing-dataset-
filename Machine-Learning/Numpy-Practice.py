
import numpy as np

#%%
arr1 = np.full((3, 3), fill_value = 3)
identity_1 = np.eye(arr1.shape[0], dtype='int64') 

#%%
# array multiplication
arr1 * identity_1

#%%
#matrix multiplication
arr1.dot(identity_1)

#%%
#Transposition
identity_1.T

#%%
#inner
np.inner(identity_1, identity_1)

#%%
#eigenvectors
np.linalg.eig(identity_1)

# %%
import numpy as np
print(np.random.randn(4,4))

#%%

import numpy as np
np.random.randint(0, 10, size = (4,4))

# %%
A = np.random.randint(0, 10, size = (4,4))
B = np.random.randint(0, 10, size = (4,4))
print(A)
print(B)
np.matmul(A, B)
A @ B 
# %%
