import numpy as np
from scipy import sparse


inter_feat = np.random.normal(5,5,[52,52,256])
inter_feat = inter_feat.astype(np.float32)
x = inter_feat.copy()
x[x<0] = 0
n = inter_feat.size
n1 = np.count_nonzero(inter_feat == 0)
n2 = np.count_nonzero(x == 0)
print(n2/n*100)
np.save('result/original',inter_feat)
np.save('result/x', x)
np.savez_compressed('result/compressed_original',inter_feat)
np.savez_compressed('result/compressed_x',x)

x_quantized = (x.copy()-x.min())/(x.max()-x.min()) *(2**8-1)
x_quantized = x_quantized.astype(np.uint8)
n3 = np.count_nonzero(x_quantized==0)
print(n3/n*100)
np.savez_compressed('result/compressed_quantized_x',x_quantized)

np.savez_compressed('result/compressed_quantized_x_c0',x_quantized[:,:,0])

x_csc = sparse.csc_matrix(x_quantized[:,:,0])
sparse.save_npz('result/csc',x_csc)

print('done')