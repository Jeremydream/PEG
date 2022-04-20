#!/usr/bin/env python
import sys
import numpy as np
import openmm
from openmm import *
from openmm.app import *
from openmm.unit import *
import jax
import jax_md
import jax.numpy as jnp
import dmff
from dmff.api import Hamiltonian
import pickle
import time
from jax import value_and_grad, jit
import optax

if __name__ == '__main__':
    restart_1 = '../peg2_peg2/params.0.pickle'
    database_1 = '../peg2_peg2/data_sr.pickle'
    ff_1 = '../peg2_peg2/forcefield.xml'
    pdb_AB_1 = PDBFile('../peg2_peg2/peg2_dimer.pdb')
    pdb_A_1 = PDBFile('../peg2_peg2/peg2.pdb')
    pdb_B_1 = PDBFile('../peg2_peg2/peg2.pdb')

    restart_2 = '../coc_peg2/params.0.pickle'
    database_2 = '../coc_peg2/data_sr.pickle' 
    ff_2 = '../coc_peg2/forcefield.xml'
    pdb_AB_2 = PDBFile('../coc_peg2/coc_peg2.pdb')
    pdb_A_2 = PDBFile('../coc_peg2/peg2.pdb')
    pdb_B_2 = PDBFile('../coc_peg2/coc.pdb')

    restart_3 = '../c_peg2/params.0.pickle' 
    database_3 = '../c_peg2/data_sr.pickle' 
    ff_3 = '../c_peg2/forcefield.xml'
    pdb_AB_3 = PDBFile('../c_peg2/ch4_peg2.pdb')
    pdb_A_3 = PDBFile('../c_peg2/peg2.pdb')
    pdb_B_3 = PDBFile('../c_peg2/ch4.pdb')

    restart_4 = '../coc_coc/params.0.pickle'
    database_4 = '../coc_coc/data_sr.pickle'  
    ff_4 = '../coc_coc/forcefield.xml'
    pdb_AB_4 = PDBFile('../coc_coc/coc_dimer.pdb')
    pdb_A_4 = PDBFile('../coc_coc/coc.pdb')
    pdb_B_4 = PDBFile('../coc_coc/coc.pdb')

    restart_5 = '../c_coc/params.0.pickle'
    database_5 = '../c_coc/data_sr.pickle'  
    ff_5 = '../c_coc/forcefield.xml'
    pdb_AB_5 = PDBFile('../c_coc/ch4_coc.pdb')
    pdb_A_5 = PDBFile('../c_coc/coc.pdb')
    pdb_B_5 = PDBFile('../c_coc/ch4.pdb')

    restart_6 = '../c_c/params.0.pickle'
    database_6 = '../c_c/data_sr.pickle'  
    ff_6 = '../c_c/forcefield.xml'
    pdb_AB_6 = PDBFile('../c_c/ch4_dimer.pdb')
    pdb_A_6 = PDBFile('../c_c/ch4.pdb')
    pdb_B_6 = PDBFile('../c_c/ch4.pdb')


    # construct total force field params
    comps = ['ex', 'es', 'pol', 'disp', 'dhf', 'tot']
    weights_comps = jnp.array([0.001, 0.001, 0.001, 0.001, 0.001, 1.0])
    with open(restart_1, 'rb') as ifile:
        params_1 = pickle.load(ifile)    
    with open(restart_2, 'rb') as ifile:
        params_2 = pickle.load(ifile)
    with open(restart_3, 'rb') as ifile:
        params_3 = pickle.load(ifile)
    with open(restart_4, 'rb') as ifile:
        params_4 = pickle.load(ifile)
    with open(restart_5, 'rb') as ifile:
        params_5 = pickle.load(ifile)
    with open(restart_6, 'rb') as ifile:
        params_6 = pickle.load(ifile)

    # load data
    with open(database_1, 'rb') as ifile:
        data_1 = pickle.load(ifile)
    with open(database_2, 'rb') as ifile:
        data_2 = pickle.load(ifile)
    with open(database_3, 'rb') as ifile:
        data_3 = pickle.load(ifile)
    with open(database_4, 'rb') as ifile:
        data_4 = pickle.load(ifile)
    with open(database_5, 'rb') as ifile:
        data_5 = pickle.load(ifile)
    with open(database_6, 'rb') as ifile:
        data_6 = pickle.load(ifile)


    # @jit
    def MSELoss(params, scan_res, ff, pdb_AB, pdb_A, pdb_B):
        '''
        The weighted mean squared error loss function
        Conducted for each scan
        '''
        # # judge the choose of forcefield
        # judge = len(jnp.array(pdb_AB.positions._value)*10)
        # if judge == 32:
        #     ff = ff_1
        # elif judge == 25:
        #     ff = ff_2
        # elif judge == 21:
        #     ff = ff_3
        # elif judge == 18:
        #     ff = ff_4
        # elif judge == 14:
        #     ff = ff_5
        # elif judge == 10:
        #     ff = ff_6

        H_AB = Hamiltonian(ff)
        H_A = Hamiltonian(ff)
        H_B = Hamiltonian(ff)
        pme_generator_AB, \
                disp_generator_AB, \
                ex_generator_AB, \
                sr_es_generator_AB, \
                sr_pol_generator_AB, \
                sr_disp_generator_AB, \
                dhf_generator_AB, \
                dmp_es_generator_AB, \
                dmp_disp_generator_AB = H_AB.getGenerators()
        pme_generator_A, \
                disp_generator_A, \
                ex_generator_A, \
                sr_es_generator_A, \
                sr_pol_generator_A, \
                sr_disp_generator_A, \
                dhf_generator_A, \
                dmp_es_generator_A, \
                dmp_disp_generator_A = H_A.getGenerators()
        pme_generator_B, \
                disp_generator_B, \
                ex_generator_B, \
                sr_es_generator_B, \
                sr_pol_generator_B, \
                sr_disp_generator_B, \
                dhf_generator_B, \
                dmp_es_generator_B, \
                dmp_disp_generator_B = H_B.getGenerators()

        rc = 15

        # get potential functions
        potentials_AB = H_AB.createPotential(pdb_AB.topology, nonbondedCutoff=rc*angstrom, ethresh=1e-4)
        pot_pme_AB, \
                pot_disp_AB, \
                pot_ex_AB, \
                pot_sr_es_AB, \
                pot_sr_pol_AB, \
                pot_sr_disp_AB, \
                pot_dhf_AB, \
                pot_dmp_es_AB, \
                pot_dmp_disp_AB = potentials_AB
        potentials_A = H_A.createPotential(pdb_A.topology, nonbondedCutoff=rc*angstrom, ethresh=1e-4)
        pot_pme_A, \
                pot_disp_A, \
                pot_ex_A, \
                pot_sr_es_A, \
                pot_sr_pol_A, \
                pot_sr_disp_A, \
                pot_dhf_A, \
                pot_dmp_es_A, \
                pot_dmp_disp_A = potentials_A
        potentials_B = H_B.createPotential(pdb_B.topology, nonbondedCutoff=rc*angstrom, ethresh=1e-4)
        pot_pme_B, \
                pot_disp_B, \
                pot_ex_B, \
                pot_sr_es_B, \
                pot_sr_pol_B, \
                pot_sr_disp_B, \
                pot_dhf_B, \
                pot_dmp_es_B, \
                pot_dmp_disp_B = potentials_B

        pos_AB0 = jnp.array(pdb_AB.positions._value) * 10
        n_atoms = len(pos_AB0)
        n_atoms_A = len(jnp.array(pdb_A.positions._value)*10)
        n_atoms_B = n_atoms - n_atoms_A
        pos_A0 = jnp.array(pdb_AB.positions._value[:n_atoms_A]) * 10
        pos_B0 = jnp.array(pdb_AB.positions._value[n_atoms_A:n_atoms]) * 10
        box = jnp.array(pdb_AB.topology.getPeriodicBoxVectors()._value) * 10
        # nn list initial allocation
        displacement_fn, shift_fn = jax_md.space.periodic_general(box, fractional_coordinates=False)
        neighbor_list_fn = jax_md.partition.neighbor_list(displacement_fn, box, rc, 0, format=jax_md.partition.OrderedSparse)
        nbr_AB = neighbor_list_fn.allocate(pos_AB0)
        nbr_A = neighbor_list_fn.allocate(pos_A0)
        nbr_B = neighbor_list_fn.allocate(pos_B0)
        pairs_AB = np.array(nbr_AB.idx.T)
        pairs_A = np.array(nbr_A.idx.T)
        pairs_B = np.array(nbr_B.idx.T)

        pairs_AB =  pairs_AB[pairs_AB[:, 0] < pairs_AB[:, 1]]
        pairs_A =  pairs_A[pairs_A[:, 0] < pairs_A[:, 1]]
        pairs_B =  pairs_B[pairs_B[:, 0] < pairs_B[:, 1]]

        E_tot_full = scan_res['tot_full']
        kT = 2.494 # 300 K = 2.494 kJ/mol
        # give the large interaction energy a weight
        weights_pts = jnp.piecewise(E_tot_full, [E_tot_full<25, E_tot_full>=25], [lambda x: jnp.array(1.0), lambda x: jnp.exp(-(x-25)/kT)])
        npts = len(weights_pts) 
        
        energies = {
                'ex': jnp.zeros(npts), 
                'es': jnp.zeros(npts), 
                'pol': jnp.zeros(npts),
                'disp': jnp.zeros(npts),
                'dhf': jnp.zeros(npts),
                'tot': jnp.zeros(npts)
                }

        # setting up params for all calculators
        params_ex = {}
        params_sr_es = {}
        params_sr_pol = {}
        params_sr_disp = {}
        params_dhf = {}
        params_dmp_es = {}  # electrostatic damping
        params_dmp_disp = {} # dispersion damping
        for k in ['B', 'mScales']:
            params_ex[k] = params[k]
            params_sr_es[k] = params[k]
            params_sr_pol[k] = params[k]
            params_sr_disp[k] = params[k]
            params_dhf[k] = params[k]
            params_dmp_es[k] = params[k]
            params_dmp_disp[k] = params[k]
        params_ex['A'] = params['A_ex']
        params_sr_es['A'] = params['A_es']
        params_sr_pol['A'] = params['A_pol']
        params_sr_disp['A'] = params['A_disp']
        params_dhf['A'] = params['A_dhf']
        # damping parameters
        params_dmp_es['Q'] = params['Q']
        params_dmp_disp['C6'] = params['C6']
        params_dmp_disp['C8'] = params['C8']
        params_dmp_disp['C10'] = params['C10']

        # calculate each points, only the short range and damping components
        for ipt in range(npts):
            # get position array
            pos_A = jnp.array(scan_res['posA'][ipt])
            pos_B = jnp.array(scan_res['posB'][ipt])
            pos_AB = jnp.concatenate([pos_A, pos_B], axis=0)
     
            #####################
            # exchange repulsion
            #####################
            E_ex_AB = pot_ex_AB(pos_AB, box, pairs_AB, params_ex)
            E_ex_A = pot_ex_A(pos_A, box, pairs_A, params_ex)
            E_ex_B = pot_ex_B(pos_B, box, pairs_B, params_ex)
            E_ex = E_ex_AB - E_ex_A - E_ex_B

            #######################
            # electrostatic + pol
            #######################
            E_dmp_es = pot_dmp_es_AB(pos_AB, box, pairs_AB, params_dmp_es) \
                     - pot_dmp_es_A(pos_A, box, pairs_A, params_dmp_es) \
                     - pot_dmp_es_B(pos_B, box, pairs_B, params_dmp_es)
            E_sr_es = pot_sr_es_AB(pos_AB, box, pairs_AB, params_sr_es) \
                    - pot_sr_es_A(pos_A, box, pairs_A, params_sr_es) \
                    - pot_sr_es_B(pos_B, box, pairs_B, params_sr_es)

            ###################################
            # polarization (induction) energy
            ###################################
            E_sr_pol = pot_sr_pol_AB(pos_AB, box, pairs_AB, params_sr_pol) \
                     - pot_sr_pol_A(pos_A, box, pairs_A, params_sr_pol) \
                     - pot_sr_pol_B(pos_B, box, pairs_B, params_sr_pol)

            #############
            # dispersion
            #############
            E_dmp_disp = pot_dmp_disp_AB(pos_AB, box, pairs_AB, params_dmp_disp) \
                       - pot_dmp_disp_A(pos_A, box, pairs_A, params_dmp_disp) \
                       - pot_dmp_disp_B(pos_B, box, pairs_B, params_dmp_disp)
            E_sr_disp = pot_sr_disp_AB(pos_AB, box, pairs_AB, params_sr_disp) \
                      - pot_sr_disp_A(pos_A, box, pairs_A, params_sr_disp) \
                      - pot_sr_disp_B(pos_B, box, pairs_B, params_sr_disp)

            ###########
            # dhf
            ###########
            E_AB_dhf = pot_dhf_AB(pos_AB, box, pairs_AB, params_dhf)
            E_A_dhf = pot_dhf_A(pos_A, box, pairs_A, params_dhf)
            E_B_dhf = pot_dhf_B(pos_B, box, pairs_B, params_dhf)
            E_dhf = E_AB_dhf - E_A_dhf - E_B_dhf

            energies['ex'] = energies['ex'].at[ipt].set(E_ex)
            energies['es'] = energies['es'].at[ipt].set(E_dmp_es + E_sr_es)
            energies['pol'] = energies['pol'].at[ipt].set(E_sr_pol)
            energies['disp'] = energies['disp'].at[ipt].set(E_dmp_disp + E_sr_disp)
            energies['dhf'] = energies['dhf'].at[ipt].set(E_dhf)
            energies['tot'] = energies['tot'].at[ipt].set(E_ex 
                                                        + E_dmp_es + E_sr_es
                                                        + E_sr_pol 
                                                        + E_dmp_disp + E_sr_disp 
                                                        + E_dhf)


        errs = jnp.zeros(len(comps))
        for ic, c in enumerate(comps):
            dE = energies[c] - scan_res[c]
            mse = dE**2 * weights_pts / jnp.sum(weights_pts)
            errs = errs.at[ic].set(jnp.sum(mse))

        return jnp.sum(weights_comps * errs)

    params = {'linear_1': params_1, 'linear_2': params_2, 'linear_3': params_3, 'linear_4': params_4, 'linear_5': params_5, 'linear_6': params_6}

    def f(params, data_1, ff_1, pdb_AB_1, pdb_A_1, pdb_B_1, 
            data_2, ff_2, pdb_AB_2, pdb_A_2, pdb_B_2, 
            data_3, ff_3, pdb_AB_3, pdb_A_3, pdb_B_3,
            data_4, ff_4, pdb_AB_4, pdb_A_4, pdb_B_4,
            data_5, ff_5, pdb_AB_5, pdb_A_5, pdb_B_5,
            data_6, ff_6, pdb_AB_6, pdb_A_6, pdb_B_6
            ):
            linear_param_1 = params['linear_1']
            linear_param_2 = params['linear_2']
            linear_param_3 = params['linear_3']
            linear_param_4 = params['linear_4']
            linear_param_5 = params['linear_5']
            linear_param_6 = params['linear_6']
            return 1.0*MSELoss(linear_param_1, data_1, ff_1, pdb_AB_1, pdb_A_1, pdb_B_1) + 0.8*MSELoss(linear_param_2, data_2, ff_2, pdb_AB_2, pdb_A_2, pdb_B_2) + 0.5*MSELoss(linear_param_3, data_3, ff_3, pdb_AB_3, pdb_A_3, pdb_B_3) + 0.1*MSELoss(linear_param_4, data_4, ff_4, pdb_AB_4, pdb_A_4, pdb_B_4) + 0.1*MSELoss(linear_param_5, data_5, ff_5, pdb_AB_5, pdb_A_5, pdb_B_5) + 0.1*MSELoss(linear_param_6, data_6, ff_6, pdb_AB_6, pdb_A_6, pdb_B_6)


    # def f(params, data_1, pdb_AB_1, pdb_A_1, pdb_B_1, 
    #         data_2, pdb_AB_2, pdb_A_2, pdb_B_2, 
    #         data_3, pdb_AB_3, pdb_A_3, pdb_B_3,
    #         data_4, pdb_AB_4, pdb_A_4, pdb_B_4,
    #         data_5, pdb_AB_5, pdb_A_5, pdb_B_5,
    #         data_6, pdb_AB_6, pdb_A_6, pdb_B_6
    #         ):
    #         linear_param_1 = params['linear_1']
    #         linear_param_2 = params['linear_2']
    #         linear_param_3 = params['linear_3']
    #         linear_param_4 = params['linear_4']
    #         linear_param_5 = params['linear_5']
    #         linear_param_6 = params['linear_6']
    #         return 1.0*MSELoss(linear_param_1, data_1, pdb_AB_1, pdb_A_1, pdb_B_1) + 0.8*MSELoss(linear_param_2, data_2, pdb_AB_2, pdb_A_2, pdb_B_2) + 0.5*MSELoss(linear_param_3, data_3, pdb_AB_3, pdb_A_3, pdb_B_3) + 0.1*MSELoss(linear_param_4, data_4, pdb_AB_4, pdb_A_4, pdb_B_4) + 0.1*MSELoss(linear_param_5, data_5, pdb_AB_5, pdb_A_5, pdb_B_5) + 0.1*MSELoss(linear_param_6, data_6, pdb_AB_6, pdb_A_6, pdb_B_6)


    err, gradients = value_and_grad(f, argnums=(0))(params, data_1['000'], ff_1, pdb_AB_1, pdb_A_1, pdb_B_1, 
                                                    data_2['000'], ff_2, pdb_AB_2, pdb_A_2, pdb_B_2, 
                                                    data_3['000'], ff_3, pdb_AB_3, pdb_A_3, pdb_B_3,
                                                    data_4['000'], ff_4, pdb_AB_4, pdb_A_4, pdb_B_4,
                                                    data_5['000'], ff_5, pdb_AB_5, pdb_A_5, pdb_B_5,
                                                    data_6['000'], ff_6, pdb_AB_6, pdb_A_6, pdb_B_6
                                                    )

    # err, gradients = value_and_grad(f, argnums=(0))(params, data_1['000'], pdb_AB_1, pdb_A_1, pdb_B_1, 
    #                                                 data_2['000'], pdb_AB_2, pdb_A_2, pdb_B_2, 
    #                                                 data_3['000'], pdb_AB_3, pdb_A_3, pdb_B_3,
    #                                                 data_4['000'], pdb_AB_4, pdb_A_4, pdb_B_4,
    #                                                 data_5['000'], pdb_AB_5, pdb_A_5, pdb_B_5,
    #                                                 data_6['000'], pdb_AB_6, pdb_A_6, pdb_B_6
    #                                                 )

    sids = np.array(list(data_1.keys()))

    # only optimize these parameters A/B
    def mask_fn(grads):
        for k in grads:
            for j in grads[k]:
                if j.startswith('A_') or j == 'B':
                    continue
                else:
                    grads[k][j] = 0.0
        return grads

    # start to do optmization
    lr = 0.001 # learning rate
    optimizer = optax.adam(lr) # The classic Adam optimiser.
    opt_state = optimizer.init(params)

    n_epochs = 1000
    for i_epoch in range(n_epochs):
        np.random.shuffle(sids)
        for sid in sids:
            loss, grads = value_and_grad(f, argnums=(0))(params, data_1[sid], ff_1, pdb_AB_1, pdb_A_1, pdb_B_1, 
                                                    data_2[sid], ff_2, pdb_AB_2, pdb_A_2, pdb_B_2, 
                                                    data_3[sid], ff_3, pdb_AB_3, pdb_A_3, pdb_B_3,
                                                    data_4[sid], ff_4, pdb_AB_4, pdb_A_4, pdb_B_4,
                                                    data_5[sid], ff_5, pdb_AB_5, pdb_A_5, pdb_B_5,
                                                    data_6[sid], ff_6, pdb_AB_6, pdb_A_6, pdb_B_6
                                                    )
            grads = mask_fn(grads)
            print(loss)
            sys.stdout.flush()
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        with open('params.test.pickle', 'wb') as ofile:
            pickle.dump(params, ofile)
