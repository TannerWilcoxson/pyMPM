import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000,
    formatter=dict(float=lambda x: "%.3g" % x))
import itertools
import time
import warnings
from scipy.sparse.linalg import gmres
from scipy.special import jv as besselj
from scipy.special import erfc
from scipy.interpolate import interp1d
from scipy.fft import fftn, fftshift, ifftn, ifftshift
from scipy.sparse.linalg import gmres,LinearOperator
from scipy.optimize import root
from scipy.sparse.linalg import lsqr

@profile
def run_MPM(posistions,box,gamma,eps_inf,xi,kmin,kmax,dk,outfile = "MPM"):# {{{
    num_particles = positions.shape[1]
    k = np.arange(kmin,kmax+dk,dk).astype("complex128")
    num_waves = len(k)
    eps_p = (eps_inf - 1/(k**2+1j*k*gamma))
    eps_in = np.ones([num_waves,num_particles])*eps_p[:,None]

    cap,dip = capacitance_tensor_spectrum(positions,box,eps_in,xi)

    np.save(outfile + "_wavenumbers.npy",k)
    np.save(outfile + "_capacitance.npy",cap)
    np.save(outfile + "_dipoles.npy",dip)
    # }}}
def plot_MPM(infile = "MPM"):# {{{
    C = np.load(infile + "_capacitance.npy")
    eps_p = np.load(infile + "_dipoles.npy")
    k = np.load(infile + "_wavenumbers.npy")
    pass# }}}
def capacitance_tensor_spectrum(pos,box,eps_p,xi,particle_dia = None,errortol = 1e-3):# {{{
    num_frames = pos.shape[0]
    num_particles = pos.shape[1]
    num_wavevectors = eps_p.shape[0]

    if particle_dia is None:
        particle_radius = 1
    if not np.iterable(particle_dia):
        eta = (4*np.pi/3 * particle_radius**3 * num_particles) / np.prod(box)
    else:
        eta = 4*np.pi/3 * np.sum(particle_radius**3) / np.prod(box)

    r_table = np.arange(1,10,.001)
    _,_,field_dip_1, field_dip_2,_,_ = real_space_table(r_table,xi)
    r_table = np.insert(r_table,0,0)

    kcut = 2*xi*np.sqrt(-np.log(errortol))
    num_grid = np.ceil(1+box*kcut/np.pi).astype(int)
    grid_spacing = box/num_grid
    num_grid_gaussian = np.ceil(-2*np.log(errortol)/np.pi)
    offset,offsetxyz = precalculations(num_grid_gaussian,grid_spacing)

    cap = np.zeros([num_frames,num_wavevectors,3,3],dtype = "complex128")
    dip = np.zeros([num_frames,num_wavevectors,num_particles,3,3], dtype = "complex128")

    for frame_idx in range(num_frames):

        ## Parallel starts here in matlab code
        for wavevec_idx in range(num_wavevectors):
            #print("k: ",wavevec_idx, num_wavevectors)
            beta = (eps_p[wavevec_idx]-1)/(eps_p[wavevec_idx] + 2)
            p_guess = np.zeros([num_particles,3]).astype('complex128')
            p_guess[:,0] = 4*np.pi*beta/(1-beta*eta)
            
            new_cap, new_dip = compute_capacitance_tensor(positions[frame_idx,:,:],eps_p[wavevec_idx],
                                                   box,p_guess,xi,field_dip_1,field_dip_2,
                                                   r_table,offset,offsetxyz,errortol = errortol)

            cap[frame_idx,wavevec_idx,:,:] = new_cap
            dip[frame_idx,wavevec_idx,:,:,:] = new_dip

    cap = np.average(cap,axis=0)
    return cap, dip# }}}
def real_space_table(r,xi):# {{{

    pi = np.pi
    exp = np.exp

    # Polynomials multiplying the exponetials
    exppolyp = -(r+2)/(32*pi**(3/2)*xi*r)
    exppolym = -(r-2)/(32*pi**(3/2)*xi*r)
    exppoly0 = 1/(16*pi**(3/2)*xi)
    
    # Polynomials multiplying the error functions
    erfpolyp = (2*xi**2*(r+2)**2 + 1)/(64*pi*xi**2*r)
    erfpolym = (2*xi**2*(r-2)**2 + 1)/(64*pi*xi**2*r)
    erfpoly0 = -(2*xi**2*r**2 + 1)/(32*pi*xi**2*r)
    
    # Regularization for overlapping particles
    regpoly = -1/(4*pi*r) + (4-r)/(16*pi)
    
    # Combine the polynomial coefficients, exponentials, and error functions
    pot_charge = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
        erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly
    
    ## Potential/Dipole or Field/Charge coupling
    
    # Polynomials multiplying the exponetials
    exppolyp = 1/(256*pi**(3/2)*xi**3*r**2)*(-6*xi**2*r**3 - 4*xi**2*r**2 + (-3+8*xi**2)*r + 2*(1-8*xi**2))
    exppolym = 1/(256*pi**(3/2)*xi**3*r**2)*(-6*xi**2*r**3 + 4*xi**2*r**2 + (-3+8*xi**2)*r - 2*(1-8*xi**2))
    exppoly0 = 3*(2*r**2*xi**2+1)/(128*pi**(3/2)*xi**3*r)
    
    # Polynomials multiplying the error functions
    erfpolyp = 1/(512*pi*xi**4*r**2)*(12*xi**4*r**4 + 32*xi**4*r**3 + 12*xi**2*r**2 - 3+64*xi**4)
    erfpolym = 1/(512*pi*xi**4*r**2)*(12*xi**4*r**4 - 32*xi**4*r**3 + 12*xi**2*r**2 - 3+64*xi**4)
    erfpoly0 = -3*(4*xi**4*r**4 + 4*xi**2*r**2 - 1)/(256*pi*xi**4*r**2)
    
    # Regularization for overlapping particles
    regpoly = -1/(4*pi*r**2) + r/(8*pi)*(1-3/8*r)
    
    # Combine the polynomial coefficients, exponentials, and error functions
    pot_dip = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
        erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly

    ## Field/Dipole coupling: I-rr component

    # Polynomials multiplying the exponentials
    exppolyp = 1/(1024*pi**(3/2)*xi**5*r**3)*(4*xi**4*r**5 - 8*xi**4*r**4 + 8*xi**2*(2-7*xi**2)*r**3 - \
        8*xi**2*(3+2*xi**2)*r**2 + (3-12*xi**2+32*xi**4)*r + 2*(3+4*xi**2-32*xi**4))
    exppolym = 1/(1024*pi**(3/2)*xi**5*r**3)*(4*xi**4*r**5 + 8*xi**4*r**4 + 8*xi**2*(2-7*xi**2)*r**3 + \
        8*xi**2*(3+2*xi**2)*r**2 + (3-12*xi**2+32*xi**4)*r - 2*(3+4*xi**2-32*xi**4))
    exppoly0 = 1/(512*pi**(3/2)*xi**5*r**2)*(-4*xi**4*r**4 - 8*xi**2*(2-9*xi**2)*r**2 - 3+36*xi**2)
    
    # Polynomials multiplying the error functions
    erfpolyp = 1/(2048*pi*xi**6*r**3)*(-8*xi**6*r**6 - 36*xi**4*(1-4*xi**2)*r**4 + 256*xi**6*r**3 - \
        18*xi**2*(1-8*xi**2)*r**2 + 3-36*xi**2+256*xi**6)
    erfpolym = 1/(2048*pi*xi**6*r**3)*(-8*xi**6*r**6 - 36*xi**4*(1-4*xi**2)*r**4 - 256*xi**6*r**3 - \
        18*xi**2*(1-8*xi**2)*r**2 + 3-36*xi**2+256*xi**6)
    erfpoly0 = 1/(1024*pi*xi**6*r**3)*(8*xi**6*r**6 + 36*xi**4*(1-4*xi**2)*r**4 + 18*xi**2*(1-8*xi**2)*r**2 - 3+36*xi**2)
    
    # Regularization for overlapping particles
    regpoly = -1/(4*pi*r**3) + 1/(4*pi)*(1-9*r/16+r**3/32)
    
    # Combine the polynomial coefficients, exponentials, and error functions
    field_dip_1 = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
        erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly
    
    
    ## Field/Dipole coupling: rr component
    
    # Polynomials multiplying the exponentials
    exppolyp = 1/(512*pi**(3/2)*xi**5*r**3)*(8*xi**4*r**5 - 16*xi**4*r**4 + 2*xi**2*(7-20*xi**2)*r**3 - \
        4*xi**2*(3-4*xi**2)*r**2 - (3-12*xi**2+32*xi**4)*r - 2*(3+4*xi**2-32*xi**4))
    exppolym = 1/(512*pi**(3/2)*xi**5*r**3)*(8*xi**4*r**5 + 16*xi**4*r**4 + 2*xi**2*(7-20*xi**2)*r**3 + \
        4*xi**2*(3-4*xi**2)*r**2 - (3-12*xi**2+32*xi**4)*r + 2*(3+4*xi**2-32*xi**4))
    exppoly0 = 1/(256*pi**(3/2)*xi**5*r**2)*(-8*xi**4*r**4 - 2*xi**2*(7-36*xi**2)*r**2 + 3-36*xi**2)
    
    # Polynomials multiplying the error functions
    erfpolyp = 1/(1024*pi*xi**6*r**3)*(-16*xi**6*r**6 - 36*xi**4*(1-4*xi**2)*r**4 + 128*xi**6*r**3 - 3+36*xi**2-256*xi**6)
    erfpolym = 1/(1024*pi*xi**6*r**3)*(-16*xi**6*r**6 - 36*xi**4*(1-4*xi**2)*r**4 - 128*xi**6*r**3 - 3+36*xi**2-256*xi**6)
    erfpoly0 = 1/(512*pi*xi**6*r**3)*(16*xi**6*r**6 + 36*xi**4*(1-4*xi**2)*r**4 + 3-36*xi**2)
    
    # Regularization for overlapping particles
    regpoly = 1/(2*pi*r**3) + 1/(4*pi)*(1-9*r/8+r**3/8)
    
    # Combine the polynomial coefficients, exponentials, and error functions
    field_dip_2 = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
    erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly



    ## Field/Dipole Force: coefficient multiplying -(mi*mj)r and -( (mj*r)mi + (mi*r)mj - 2(mi*r)(mj*r)r )

    # Polynomials multiplying the exponentials
    exppolyp = 3/(1024*pi**(3/2)*xi**5*r**4)*(4*xi**4*r**5 - 8*xi**4*r**4 + 4*xi**2*(1-2*xi**2)*r**3 + 16*xi**4*r**2 - (3-12*xi**2+32*xi**4)*r - 2*(3+4*xi**2-32*xi**4))
    exppolym = 3/(1024*pi**(3/2)*xi**5*r**4)*(4*xi**4*r**5 + 8*xi**4*r**4 + 4*xi**2*(1-2*xi**2)*r**3 - 16*xi**4*r**2 - (3-12*xi**2+32*xi**4)*r + 2*(3+4*xi**2-32*xi**4))
    exppoly0 = 3/(512*pi**(3/2)*xi**5*r**3)*(-4*xi**4*r**4 - 4*xi**2*(1-6*xi**2)*r**2 + 3-36*xi**2)
    
    # Polynomials multiplying the error functions
    erfpolyp = 3/(2048*pi*xi**6*r**4)*(-8*xi**6*r**6 - 12*xi**4*(1-4*xi**2)*r**4 + 6*xi**2*(1-8*xi**2)*r**2 - 3+36*xi**2-256*xi**6)
    erfpolym = 3/(2048*pi*xi**6*r**4)*(-8*xi**6*r**6 - 12*xi**4*(1-4*xi**2)*r**4 + 6*xi**2*(1-8*xi**2)*r**2 - 3+36*xi**2-256*xi**6)
    erfpoly0 = 3/(1024*pi*xi**6*r**4)*(8*xi**6*r**6 + 12*xi**4*(1-4*xi**2)*r**4 - 6*xi**2*(1-8*xi**2)*r**2 + 3-36*xi**2)
    
    # Regularization for overlapping particles
    regpoly = 3/(4*pi*r**4) - 3/(64*pi)*(3-r**2/2)
    
    # Combine the polynomial coefficients, exponentials, and error functions
    field_dip_force_1 = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
        erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly
    
    ## Field/Dipole Force from:  coefficient multiplying -(mi*r)(mj*r)r
    
    # Polynomials multiplying the exponentials
    exppolyp = 9/(1024*pi**(3/2)*xi**5*r**4)*(4*xi**4*r**5 - 8*xi**4*r**4 + 8*xi**4*r**3 + 8*xi**2*(1-2*xi**2)*r**2 + (3-12*xi**2+32*xi**4)*r + 2*(3+4*xi**2-32*xi**4))
    exppolym = 9/(1024*pi**(3/2)*xi**5*r**4)*(4*xi**4*r**5 + 8*xi**4*r**4 + 8*xi**4*r**3 - 8*xi**2*(1-2*xi**2)*r**2 + (3-12*xi**2+32*xi**4)*r - 2*(3+4*xi**2-32*xi**4))
    exppoly0 = 9/(512*pi**(3/2)*xi**5*r**3)*(-4*xi**4*r**4 + 8*xi**4*r**2 - 3+36*xi**2)
    
    # Polynomials multiplying the error functions
    erfpolyp = 9/(2048*pi*xi**6*r**4)*(-8*xi**6*r**6 - 4*xi**4*(1-4*xi**2)*r**4 - 2*xi**2*(1-8*xi**2)*r**2 + 3-36*xi**2+256*xi**6)
    erfpolym = 9/(2048*pi*xi**6*r**4)*(-8*xi**6*r**6 - 4*xi**4*(1-4*xi**2)*r**4 - 2*xi**2*(1-8*xi**2)*r**2 + 3-36*xi**2+256*xi**6)
    erfpoly0 = 9/(1024*pi*xi**6*r**4)*(8*xi**6*r**6 + 4*xi**4*(1-4*xi**2)*r**4 + 2*xi**2*(1-8*xi**2)*r**2 - 3+36*xi**2)
    
    # Regularization for overlapping particles
    regpoly = -9/(4*pi*r**4) - 9/(64*pi)*(1-r**2/2)
    
    # Combine the polynomial coefficients, exponentials, and error functions
    field_dip_force_2 = exppolyp*exp(-(r+2)**2*xi**2) + exppolym*exp(-(r-2)**2*xi**2) + exppoly0*exp(-r**2*xi**2) + \
    erfpolyp*erfc((r+2)*xi) + erfpolym*erfc((r-2)*xi) + erfpoly0*erfc(r*xi) + (r < 2)*regpoly


    ## Self terms
    
    # Potential/charge
    self = (1-exp(-4*xi**2))/(8*pi**(3/2)*xi) + erfc(2*xi)/(4*pi)
    pot_charge = np.insert(pot_charge,0,self)
    
    # Potential/dipole or field/charge
    pot_dip = np.insert(pot_dip,0,0)
    
    # Field/dipole
    self = (-1+6*xi**2+(1-2*xi**2)*exp(-4*xi**2))/(16*pi**(3/2)*xi**3) + erfc(2*xi)/(4*pi)
    field_dip_1 = np.insert(field_dip_1,0,self)
    field_dip_2 = np.insert(field_dip_2,0,self)
    
    # Field/dipole force
    field_dip_force_1 = np.insert(field_dip_force_1,0,0)
    field_dip_force_2 = np.insert(field_dip_force_2,0,0)

    return pot_charge,pot_dip,field_dip_1,field_dip_2,field_dip_force_1,field_dip_force_2# }}}
def compute_capacitance_tensor(positions, lambda_p, box, dip_guess, xi, # {{{
                               dipoletable1, dipoletable2, rtable, offset,
                               offsetxyz,errortol):


    num_particles = positions.shape[0]

    H0 = np.identity(3)
    cap = np.zeros([3,3],dtype = "complex128")
    dip = np.zeros([num_particles,3,3],dtype = "complex128")

    rc = np.sqrt(-np.log(errortol))/xi
    kcut = 2*xi**2*rc
    num_grid = np.ceil(1+box*kcut/np.pi).astype(int)
    grid_spacing = box/num_grid
    num_grid_gaussian = np.ceil(-2*np.log(errortol)/np.pi)

    eta = num_grid_gaussian * (grid_spacing*xi)**2/np.pi

    if np.any(rc > box/2):
        raise Exception(f"Real space cutoff ({rc:.3f}) larger than half the box length.")

    p1,p2 = gen_neighbor_list(positions,box,rc,half=False)

    for dim in range(3):

        dip_dim = magnetic_dipole(positions,lambda_p,H0[dim],box,p1,p2,
                                num_grid,grid_spacing,num_grid_gaussian,
                                xi,eta,rc,dip_guess,offset,offsetxyz,
                                dipoletable1,dipoletable2,rtable,errortol)
        dip[:,:,dim] = dip_dim
        cap[dim,:] = np.average(dip_dim,axis = 0)

        dip_guess = dip_dim[:,[2,0,1]]
    return cap,dip
# }}}
def magnetic_dipole(positions,lambda_p,H,box,p1,p2,num_grid,grid_spacing,num_grid_gaussian,# {{{
        xi, eta, rc, dip_guess, offset, offsetxyz, dip_perp, dip_para, rvals, errortol):


    num_particles = positions.shape[0]
    restart = min([3*num_particles,10])
    maxit = min([3*num_particles,100])

    dip_guess = dip_guess.flatten()

    #---- k value calculations ----{{{
    warnings.filterwarnings('ignore')

    Kx = np.arange(-np.ceil((num_grid[0]-1)/2),np.floor((num_grid[0] - 1)/2)+1) * 2*np.pi/box[0]
    Ky = np.arange(-np.ceil((num_grid[1]-1)/2),np.floor((num_grid[1] - 1)/2)+1) * 2*np.pi/box[1]
    Kz = np.arange(-np.ceil((num_grid[2]-1)/2),np.floor((num_grid[2] - 1)/2)+1) * 2*np.pi/box[2]

    k0x = np.argwhere(Kx == 0)
    k0y = np.argwhere(Ky == 0)
    k0z = np.argwhere(Kz == 0)

    kx,ky,kz = np.meshgrid(Kx,Ky,Kz,indexing='ij')
    k = np.concatenate([kx[:,:,:,None],ky[:,:,:,None],kz[:,:,:,None]],axis = -1)

    k0_ind = np.array([k0x,k0y,k0z])

    ksq = k**2
    ksqsm = np.sum(ksq,axis = -1)
    kmag = np.sqrt(ksqsm)
    khat = k/kmag[:,:,:,None]
    khat[*k0_ind] = 0

    etaksq  = np.sum(ksq*(1-eta),axis = -1)
    Htilde_coeff = 9*np.pi/(2*kmag) * besselj(1+1/2,kmag)**2 * np.exp(-etaksq/(4*xi**2)) / ksqsm
    Htilde_coeff[*k0_ind] = 0

    kvals = dict()
    kvals["k"] = k
    kvals["ksq"] = ksq
    kvals["ksqsm"] = ksqsm
    kvals["kmag"] = kmag
    kvals["khat"] = khat
    kvals["k0_ind"] = k0_ind
    kvals["Htilde_coeff"] = Htilde_coeff
    warnings.filterwarnings('default')# }}}

    #Preallocations
    Hspace = np.zeros(np.append(num_grid,3)).astype('complex128')

    #Guess
    Hrep = np.array(H.tolist()*num_particles)

    def solve(dip):
        H = magnetic_field(positions,dip,lambda_p, box, p1, p2, num_grid, 
                       grid_spacing, num_grid_gaussian, xi, eta, rc, offset,
                       offsetxyz, dip_perp, dip_para, rvals, kvals,Hspace)
        ret = H.flatten()
        return ret


    dip = dip_guess.reshape(num_particles,3)
    solve = LinearOperator(2*[3*num_particles], matvec = solve,dtype = "complex128")

    restart = min([num_particles*3,10])
    maxiter = min([num_particles*3,100])
    dip,info = gmres(solve,Hrep,x0 = dip_guess.reshape(3*num_particles),atol=errortol,
                            restart = restart, maxiter = maxiter)
    dip = dip.reshape(num_particles,3)

    return dip

    # }}}
def magnetic_field(positions, dipoles, lambda_p, box, p1, p2, num_grid, #{{{
                   grid_size, num_grid_gaussian, xi, eta, rc, offset, 
                   offsetxyz, dip_perp, dip_para,rvals,kvals,Hspace):

    num_particles = positions.shape[0]
    dipoles = dipoles.reshape(num_particles,3)

    H = spread(positions, dipoles, num_grid, grid_size, xi,eta,num_grid_gaussian,offset,offsetxyz,Hspace)

    Hx,Hy,Hz = H.T
    Hx,Hy,Hz = Hx.T,Hy.T,Hz.T

    fHx = fftn(Hx,overwrite_x=True)
    fHx = fftshift(Hx)[:,:,:,None]

    fHy = fftn(Hy,overwrite_x=True)
    fHy = fftshift(Hy)[:,:,:,None]

    fHz = fftn(Hz,overwrite_x=True)
    fHz = fftshift(Hz)[:,:,:,None]

    fH = np.concatenate([fHx,fHy,fHz],axis = -1,out=Hspace)

    fHtilde = scale(fH,kvals,num_grid,xi,eta)
    fHtildex = fHtilde[:,:,:,0]

    Htildex = ifftn(ifftshift(fHtilde[:,:,:,0]))[:,:,:,None]
    Htildey = ifftn(ifftshift(fHtilde[:,:,:,1]))[:,:,:,None]
    Htildez = ifftn(ifftshift(fHtilde[:,:,:,2]))[:,:,:,None]
    Htilde = np.concatenate([Htildex,Htildey,Htildez],axis = -1,out=Hspace)

    Hk = contract(positions,num_grid,grid_size,xi,eta,num_grid_gaussian,Htilde,offset,offsetxyz)
    Hr = real_space(positions,dipoles,lambda_p, box, p1,p2,rc,dip_perp,dip_para,rvals)
    
    ret = Hk + Hr

    return Hk + Hr
    # }}}
def spread(positions,dip,num_grid,grid_spacing,xi,eta,num_grid_gaussian,offset,offsetxyz,H):# {{{
    num_particles = positions.shape[0]
    num_spread = num_particles*len(offset)
    H[:,:,:,:] = 0


    grid_idxs = np.round(positions/grid_spacing).astype(int)
    particle_grid_dist = grid_idxs*grid_spacing - positions
    grid_effect_idxs = (grid_idxs[:,None,:] + offset[None,:,:] - 1) % num_grid
    grid_effect_dist = (particle_grid_dist[:,None,:] + offsetxyz[None,:,:])

    grid_effect_div_eta = np.sum(grid_effect_dist**2/eta,axis = -1)

    Hcoef = (2*xi**2/np.pi)**(3/2)*np.sqrt(1/np.prod(eta))*np.exp(-2*xi**2*grid_effect_div_eta)
    Hspread = Hcoef[:,:,None]*dip[:,None,:]

    grid_effect_idxs = grid_effect_idxs.reshape(num_spread,3)
    Hspread = Hspread.reshape(num_spread,3)

    np.add.at(H,tuple(grid_effect_idxs.T),Hspread)

    return H# }}}
def scale(fH, kvals, num_grid, xi, eta):# {{{

    k = kvals["k"]
    ksq = kvals["ksq"]
    ksqsm = kvals["ksqsm"]
    kmag = kvals["kmag"]
    khat = kvals["khat"]
    k0_ind = kvals["k0_ind"]
    Htilde_coeff = kvals["Htilde_coeff"]


    np.multiply(fH,khat,out=fH)
    fH = fH.T
    sum = Htilde_coeff*(fH[0]+fH[1]+fH[2]).T
    fH = fH.T
    np.multiply(khat,sum[:,:,:,None],out=fH)
    return fH
# }}}
def contract(positions,num_grid,grid_spacing,xi,eta,num_grid_gaussian,Htilde,offset,offsetxyz):# {{{
    num_particles = positions.shape[0]
    num_spread = num_particles*len(offset)
    Hk = np.zeros([num_particles,3],dtype="complex128")

    grid_idxs = np.round(positions/grid_spacing).astype(int)
    particle_grid_dist = grid_idxs*grid_spacing - positions
    grid_effect_idxs = (grid_idxs[:,None,:] + offset[None,:,:] - 1) % num_grid
    grid_effect_idxs = grid_effect_idxs.reshape(num_spread,3)

    grid_effect_dist = (particle_grid_dist[:,None,:] + offsetxyz[None,:,:])
    grid_effect_div_eta = np.sum(grid_effect_dist**2/eta,axis = -1)

    Hcoef = (2*xi**2/np.pi)**(3/2)*np.sqrt(1/np.prod(eta))*np.exp(-2*xi**2*grid_effect_div_eta)
    Hcoef *= np.prod(grid_spacing)
    Hcoef = Hcoef.reshape(num_spread)
    particle_index = np.repeat(np.arange(num_particles),len(offset))


    np.add.at(Hk,particle_index,Hcoef[:,None]*Htilde[*grid_effect_idxs.T])
    return Hk
# }}}
def real_space(positions, dip, lambda_p, box, p1, p2, rc, dip_perp, dip_para,rvals):# {{{
    Hr = -3/(4*np.pi*(1-lambda_p[:,None])) * dip

    Hr = Hr + dip*dip_perp[0]

    r = positions[p1] - positions[p2]
    r = r-box*(2*r/box).astype(int)
    d = np.sqrt(np.sum(r**2,axis = -1))

    cutoff_flags = d<rc
    d = d[cutoff_flags]
    r = r[cutoff_flags]
    r = r[:,:]/d[:,None]

    p1 = p1[cutoff_flags]
    p2 = p2[cutoff_flags]

    dip_p2 = dip[p2]
    int_perp = interp1d(rvals,dip_perp)
    int_para = interp1d(rvals,dip_para)

    perp = int_perp(d)
    para = int_para(d)

    r_dip_p2 = np.sum(dip_p2*r,axis = -1)
    np.add.at(Hr,p1,perp[:,None]*(dip_p2 - r*r_dip_p2[:,None]) + para[:,None]*r*r_dip_p2[:,None])
    return Hr
# }}}
def precalculations(num_grid_gaussian,h):# {{{
    off = int(num_grid_gaussian/2)
    min_off = -off
    max_off = off+1
    offset = []
    for x in range(min_off,max_off):
        for y in range(min_off,max_off):
            for z in range(min_off,max_off):
                offset.append([x,y,z])
    offset = np.array(offset)[:,[2,1,0]]
    offsetxyz = offset*h
    return offset, offsetxyz# }}}
def gen_neighbor_list(positions,box_length,cutoff,half=True):#{{{
    if np.any(positions < 0):
        positions = np.copy(positions)
        positions += box_length/2
    numBoxes = (box_length/cutoff).astype(int)
    if np.any(numBoxes < 3):
        raise Exception(f"Neighbor cutoff yields less than three cells:\n    box: {box_length}\n   cutoff: {cutoff}")
    cutoff = box_length/numBoxes
    if len(positions.shape) == 2:
        numFrames = 1
        numParticles = positions.shape[0]
        dims = positions.shape[1]
    elif len(positions.shape) == 3:
        numFrames = positions.shape[0]
        numParticles = positions.shape[1]
        dims = positions.shape[2]
    indices = (positions/cutoff).astype(int)
    if not np.iterable(numBoxes):
        numBoxes = np.array([numBoxes]*dims)
    boxes = {}
    count = np.zeros(numBoxes)
    for i,idx in enumerate(indices):
        index = tuple(idx.tolist())
        if index not in boxes:
            boxes[index] = list()
        boxes[index].append(int(i))
        count[index] += 1
    P1,P2 = [],[]
    offset = np.array([np.arange(-1,2) for i in range(dims)]).T
    for p1,idx in enumerate(indices):
        offsets = (idx+offset)%numBoxes
        for off in itertools.product(*offsets.T):
            if count[off] == 0:
                continue
            p2s = boxes[off]
            P1 += [p1]*len(p2s)
            P2 += (p2s)
    P1 = np.array(P1)
    P2 = np.array(P2)
    if half:
        flags = P1 < P2
    else:
        flags = P1 != P2
    P1 = P1[flags]
    P2 = P2[flags]
    return P1,P2
    #}}}

if __name__ == "__main__":
    positions = np.array(
                [[[-1,2.8,0],
                 [-4.6,2.3,0],
                 [-3.05,1,0],

                 [-1.35,-0.45,0],
                 [0.4,-1.4,0],

                 [4.3,1.2,0],
                 [2.65,0.1,0],
                 [4.5,-1,0],

                 [-4.2,-1.05,0],
                 [-2.75,-2.5,0]]])

    box = np.array([30,30,30])
    
    run_MPM(positions,box,0.05,2,0.5,0.01,0.8,0.01)
    plot_MPM()
