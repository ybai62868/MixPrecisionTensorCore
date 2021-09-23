import tvm
import tvm.testing
from tvm import te
import numpy as np

tgt = tvm.target.Target(target="llvm")

# n = te.var("n")
# A = te.placeholder((n, ), name = "A", dtype = "float32")
# B = te.placeholder((n, ), name = "B", dtype = "float32")
# C = te.compute(A.shape, lambda i: A[i] + B[i], name = "C")

# s = te.create_schedule(C.op)
# print(tvm.lower(s, [A, B, C], simple_mode = True))


# for ( int i = 0;i < n;i++ ) {
#     C[i] = A[i] + B[i];
# }


# n = te.var("n")
# m = te.var("m")
# l = te.var("l")

# A = te.placeholder((n, l), name = "A", dtype = "float32")
# B = te.placeholder((l, m), name = "B", dtype = "float32")
# C = te.placeholder((n, m), name = "C", dtype = "float32")

# k = te.reduce_axis((0, l), name = "k")
# matmul = te.compute(
#     (n, m),
#     lambda i, j: te.sum(A[i, k] * B[k, j], axis = k),
#     name = "matmul"
# )
# gemm = te.compute(
#     (n, m),
#     lambda i, j: matmul[i, j] + C[i, j],
#     name = "gemm"
# )

# s = te.create_schedule(gemm.op)
# print(tvm.lower(s, [A, B, C], simple_mode = True))

# n = te.var("n")
# Input = te.placeholder((n, n), name = "Input")
# Filter = te.placeholder((5, 5), name = "Filter")
# di = te.reduce_axis((0, 5), name = "di")
# dj = te.reduce_axis((0, 5), name = "dj")

# Output = te.compute(
#     (n - 4, n - 4),
#     lambda i, j: te.sum(Input[i + di, j + dj] * Filter[di, dj], axis = [di, dj]),
#     name = "Output",
# )

# s = te.create_schedule(Output.op)
# print(tvm.lower(s, [Input, Filter, Output], simple_mode = True))


# n = te.var("n")
# m = te.var("m")
# A0 = te.placeholder((m, n), name = "A0")
# A1 = te.placeholder((m, n), name = "A1")
# B0, B1 = te.compute((m, n), lambda i, j: (A0[i, j] * 2, A1[i, j] + 5), name = "B")
# s = te.create_schedule(B0.op) # B1.op
# print(tvm.lower(s, [A0, A1, B0, B1], simple_mode = True))



n = te.var("n")
m = te.var("m")
A0 = te.placeholder((m, n), name = "A0")
A1 = te.placeholder((m, n), name = "A1")
B0 = te.compute((m, n), lambda i, j: A0[i, j] * 2, name = "B0")
B1 = te.compute((m, n), lambda i, j: A1[i, j] + 5, name = "B1")
s0 = te.create_schedule(B0.op) 
print(tvm.lower(s0, [A0, A1, B0, B1], simple_mode = True))
s1 = te.create_schedule(B1.op)
print(tvm.lower(s1, [A0, A1, B0, B1], simple_mode = True))