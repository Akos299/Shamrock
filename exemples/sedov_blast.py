import shamrock
import matplotlib.pyplot as plt

gamma = 5./3.
rho_g = 1
pmass = -1

#Nx = 200
#Ny = 230
#Nz = 245

Nx = 100
Ny = 130
Nz = 145



ctx = shamrock.Context()
ctx.pdata_layout_new()

sim = shamrock.BasicGasSPH(ctx)
sim.setup_fields()

#start the scheduler
ctx.init_sched(int(1e7),1)

setup = shamrock.SetupSPH(kernel = "M4", precision = "double")
setup.init(ctx)

(xs,ys,zs) = setup.get_box_dim(1,Nx,Ny,Nz)
dr = 1/xs
(xs,ys,zs) = setup.get_box_dim(dr,Nx,Ny,Nz)

ctx.set_coord_domain_bound((-xs/2,-ys/2,-zs/2),(xs/2,ys/2,zs/2))

setup.set_boundaries("periodic")

setup.add_particules_fcc(ctx,dr, (-xs/2,-ys/2,-zs/2),(xs/2,ys/2,zs/2))

rinj = 0.01
u_inj = 100

xc,yc,zc = setup.get_closest_part_to(ctx,(0,0,0))

del sim
del setup
del ctx



ctx = shamrock.Context()
ctx.pdata_layout_new()

sim = shamrock.BasicGasSPH(ctx)
sim.setup_fields()

#start the scheduler
ctx.init_sched(int(1e7),1)

setup = shamrock.SetupSPH(kernel = "M4", precision = "double")
setup.init(ctx)

(xs,ys,zs) = setup.get_box_dim(1,Nx,Ny,Nz)
dr = 1/xs
(xs,ys,zs) = setup.get_box_dim(dr,Nx,Ny,Nz)

bmin = (-xs/2-xc,-ys/2-yc,-zs/2-zc)
bmax = (xs/2-xc,ys/2-yc,zs/2-zc)

ctx.set_coord_domain_bound(bmin,bmax)

setup.set_boundaries("periodic")

setup.add_particules_fcc(ctx,dr, bmin,bmax)

rinj = 0.01
u_inj = 100

setup.set_value_in_box(ctx, "f64", 0.005, "uint", bmin,bmax)
setup.set_value_in_sphere(ctx, "f64", u_inj, "uint",(0,0,0),rinj)



vol_b = xs*ys*zs

totmass = (rho_g*vol_b)

print("Total mass :", totmass)

setup.set_total_mass(totmass)

pmass = setup.get_part_mass()

print("Current part mass :", pmass)

#for it in range(5):
#    setup.update_smoothing_lenght(ctx)



del setup



sim.set_cfl_cour(1e-1)
sim.set_cfl_force(0.3)





print("Current part mass :", pmass)


sim.set_particle_mass(pmass)

for i in range(9):
    sim.evolve(5e-4, False, False, "", False)

for i in range(200):
    sim.evolve(5e-4, True, True, "dump_"+str(i)+".vtk", True)