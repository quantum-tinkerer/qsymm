import numpy as np
import kwant
from kwant._common import get_parameters

import qsymm

def builder_to_model(syst, momenta=None):
    """Convert a kwant.Builder to a qsymm.Model"""
    def term_to_model(d, par, matrix):
        loc = dict(k=k, d=d)
        # XXX: put matrix in the correct block for the full Hamiltonian matrix
        return qsymm.Model({qsymm.sympify(par + '* exp(1j * k.dot(d))', locals=loc): matrix},
                           momenta=momenta)

    def hopping_to_term(hop, value):
        site1, site2 = hop
        d = np.array(site2.pos - site1.pos)
        slice1, slice2 = slices[to_fd(site1)], slices[to_fd(site2)]
        if callable(value):
            return sum(term_to_model(d, par, set_block(slice1, slice2, val))
                       for par, val in function_to_terms(hop, value))
        else:
            matrix = set_block(slice1, slice2, value)
            return term_to_model(d, '1', matrix)

    def onsite_to_term(site, value):
        d = np.zeros((len(k), ))
        slice1 = slices[to_fd(site)]
        if callable(value):
            return sum(term_to_model(d, par, set_block(slice1, slice1, val))
                       for par, val in function_to_terms(site, value))
        else:
            return term_to_model(d, '1', set_block(slice1, slice1, value))

    def function_to_terms(site_or_hop, value):
        assert callable(value)
        parameters = get_parameters(value)
        if isinstance(site_or_hop, kwant.builder.Site):
            parameters = parameters[1:]
            site_or_hop = (site_or_hop,)
        else:
            parameters = parameters[2:]
        h_0 = value(*site_or_hop, *((0,) * len(parameters)))
        all_args = np.eye(len(parameters))
        
        terms = []
        for p, args in zip(parameters, all_args):
            terms.append((p, value(*site_or_hop, *args) - h_0))
        return terms + [('1', h_0)]

    def orbital_slices(syst):
        orbital_slices = {}
        start_orb = 0

        for site in syst.sites():
            try:
                n = site.family.norbs
            except:
                raise ValueError('norbs must be provided for every lattice.')
            orbital_slices[site] = slice(start_orb, start_orb + n)
            start_orb += n
        return orbital_slices, start_orb
    
    def set_block(slice1, slice2, val):
        matrix = np.zeros((N, N), dtype=complex)
        matrix[slice1, slice2] = val
        return matrix
    
    dim = len(syst.symmetry.periods)
    to_fd = syst.symmetry.to_fd
    if momenta is None:
        momenta = ['k_x', 'k_y', 'k_z']
    k = qsymm.sympify('[' + ', '.join(momenta[:dim]) + ']')
    
    slices, N = orbital_slices(syst)
    
    one_way_hoppings = [hopping_to_term(hop, value) for hop, value in syst.hopping_value_pairs()]
    hoppings = one_way_hoppings + [term.T().conj() for term in one_way_hoppings]
    
    onsites = [onsite_to_term(site, value) for site, value in syst.site_value_pairs()]
    
    return sum(onsites) + sum(hoppings)