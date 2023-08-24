import shamrock
import numpy as np
import matplotlib.pyplot as plt 
import os


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_AMRZeus(
    context = ctx, 
    vector_type = "f64_3",
    grid_repr = "i64_3")

model.init_scheduler(int(1e7),1)

multx = 1
multy = 1
multz = 1

sz = 1 << 1
base = 128 
model.make_base_grid((0,0,0),(sz,sz,sz),(base*multx,base*multy,base*multz))

model.dump_vtk("test.vtk")

model.evolve_once(0,0.1)
#model.evolve_once(0,0.1)

for i in range(10):
    model.evolve_once(0,0.1)
    model.dump_vtk("test"+str(i)+".vtk")