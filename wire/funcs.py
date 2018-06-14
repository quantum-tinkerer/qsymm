# 1. Standard library imports
from copy import copy, deepcopy
from functools import partial
import operator
from types import SimpleNamespace

# 2. External package imports
import kwant
from kwant.continuum.discretizer import discretize
from kwant.digest import uniform
import numpy as np
import scipy.constants

# 3. Internal imports
from .combine import combine


# Parameters taken from arXiv:1204.2792
# All constant parameters, mostly fundamental constants, in a SimpleNamespace.
constants = SimpleNamespace(
    m_eff=0.015 * scipy.constants.m_e,  # effective mass in kg
    hbar=scipy.constants.hbar,
    m_e=scipy.constants.m_e,
    eV=scipy.constants.eV,
    e=scipy.constants.e,
    c=1e18 / (scipy.constants.eV * 1e-3),  # to get to meV * nm^2
    mu_B=scipy.constants.physical_constants['Bohr magneton in eV/T'][0] * 1e3,
    )

constants.t = (constants.hbar ** 2 / (2 * constants.m_eff)) * constants.c


# Hamiltonian and system definition
def discretized_hamiltonian(a, delta_barrier=True, as_lead=False,
                            rotate_spin_orbit=False, intrinsic_sc=False):

    SO_z = "alpha * (k_y * kron(sigma_x, sigma_z) - k_x * kron(sigma_y, sigma_z)) + "
    SO_rotated = ("alpha * k_x * (kron(sigma_z, sigma_z) * cos(theta_SO) - kron(sigma_y, sigma_z) * sin(theta_SO)) + "
                  "alpha * kron(sigma_x, sigma_z) * (k_y * sin(theta_SO) - k_z * cos(theta_SO)) + ")

    ham = ("(0.5 * hbar**2 * (k_x**2 + k_y**2 + k_z**2) / m_eff * c - mu + V) * kron(sigma_0, sigma_z) + "
           + (SO_rotated if rotate_spin_orbit else SO_z) +
           "0.5 * g * mu_B * (B_x * kron(sigma_x, sigma_0) + B_y * kron(sigma_y, sigma_0) + B_z * kron(sigma_z, sigma_0)) + "
           "Delta * kron(sigma_0, sigma_x)")

    lead = {'mu': 'mu_lead'} if as_lead else {}

    subst_sm = {'Delta': 0, 'V': 'V(x, y, z)', **lead}

    if delta_barrier:
        subst_barrier = {'mu': 'mu - V_barrier', **subst_sm}
    elif not as_lead:
        # If as_lead, there cannot be a function dependent on x.
        subst_sm['mu'] = 'mu - V_barrier(x, z, V_0)'
        subst_barrier = subst_sm

    subst_sc = {'g': 0, 'alpha': 0, 'mu': 'mu_sc', 'V': 0}
    subst_interface = {'c': 'c * c_tunnel', 'alpha': 0, 'V': 0, **lead}

    templ_sm = discretize(ham, locals=subst_sm, grid_spacing=a)
    templ_sc = discretize(ham, locals=subst_sc, grid_spacing=a)
    templ_interface = discretize(ham, locals=subst_interface, grid_spacing=a)
    templ_barrier = discretize(ham, locals=subst_barrier, grid_spacing=a)

    if intrinsic_sc:
        subst_sm.pop('Delta')  # The Hamiltonian is the same except with Delta
        templ_sc = discretize(ham, locals=subst_sm, grid_spacing=a)
        return templ_sm, templ_sc, templ_barrier

    return templ_sm, templ_sc, templ_interface, templ_barrier


def add_disorder_to_template(template):
    # Only works with particle-hole + spin DOF or only spin.
    template = deepcopy(template)  # Needed because kwant.Builder is mutable
    s0 = np.eye(2, dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    s0sz = np.kron(s0, sz)
    norbs = template.lattice.norbs
    mat = s0sz if norbs == 4 else s0

    def onsite_disorder(site, disorder, salt):
        return disorder * (uniform(repr(site), repr(salt)) - .5) * mat

    for site, onsite in template.site_value_pairs():
        onsite = template[site]
        template[site] = combine(onsite, onsite_disorder, operator.add, 1)

    return template


def apply_peierls_to_template(template, xyz_offset=(0, 0, 0)):
    """Adds p.orbital argument to the hopping functions."""
    template = deepcopy(template)  # Needed because kwant.Builder is mutable
    x0, y0, z0 = xyz_offset
    lat = template.lattice
    a = np.max(lat.prim_vecs)  # lattice contant

    def phase(site1, site2, B_x, B_y, B_z, orbital, e, hbar):
        if orbital:
            x, y, z = site1.tag
            direction = site1.tag - site2.tag
            A = [B_y * (z - z0) - B_z * (y - y0), 0, B_x * (y - y0)]
            A = np.dot(A, direction) * a**2 * 1e-18 * e / hbar
            phase = np.exp(-1j * A)
            if lat.norbs == 2:  # No PH degrees of freedom
                return phase
            elif lat.norbs == 4:
                return np.array([phase, phase.conj(), phase, phase.conj()],
                                dtype='complex128')
        else:  # No orbital phase
            return 1

    for (site1, site2), hop in template.hopping_value_pairs():
        template[site1, site2] = combine(hop, phase, operator.mul, 2)
    return template


def get_offset(shape, start, lat):
    a = np.max(lat.prim_vecs)
    coords = [site.pos for site in lat.shape(shape, start)()]
    xyz_offset = np.mean(coords, axis=0)
    return xyz_offset


# Shape functions

def square_sector(r_out, r_in=0, L=1, L0=0, coverage_angle=360, angle=0, a=10):
    """Returns the shape function and start coords of a wire
    with a square cross section, for -r_out <= x, y < r_out.

    Parameters
    ----------
    r_out : int
        Outer radius in nm.
    r_in : int
        Inner radius in nm.
    L : int
        Length of wire from L0 in nm, -1 if infinite in x-direction.
    L0 : int
        Start position in x.
    coverage_angle : ignored
        Ignored variable, to have same arguments as cylinder_sector.
    angle : ignored
        Ignored variable, to have same arguments as cylinder_sector.
    a : int
        Discretization constant in nm.

    Returns
    -------
    (shape_func, *(start_coords))
    """
    r_in /= 2
    r_out /= 2
    if r_in > 0:
        def shape(site):
            try:
                x, y, z = site.pos
            except AttributeError:
                x, y, z = site
            shape_yz = -r_in <= y < r_in and r_in <= z < r_out
            return (shape_yz and L0 <= x < L) if L > 0 else shape_yz
        return shape, np.array([L / a - 1, 0, r_in / a + 1], dtype=int)
    else:
        def shape(site):
            try:
                x, y, z = site.pos
            except AttributeError:
                x, y, z = site
            shape_yz = -r_out <= y < r_out and -r_out <= z < r_out
            return (shape_yz and L0 <= x < L) if L > 0 else shape_yz
        return shape, (int((L - a) / a), 0, 0)


def cylinder_sector(r_out, r_in=0, L=1, L0=0, coverage_angle=360, angle=0, a=10):
    """Returns the shape function and start coords for a wire with
    as cylindrical cross section.

    Parameters
    ----------
    r_out : int
        Outer radius in nm.
    r_in : int, optional
        Inner radius in nm.
    L : int, optional
        Length of wire from L0 in nm, -1 if infinite in x-direction.
    L0 : int, optional
        Start position in x.
    coverage_angle : int, optional
        Coverage angle in degrees.
    angle : int, optional
        Angle of tilting from top in degrees.
    a : int, optional
        Discretization constant in nm.

    Returns
    -------
    (shape_func, *(start_coords))
    """
    coverage_angle *= np.pi / 360
    angle *= np.pi / 180
    r_out_sq, r_in_sq = r_out**2, r_in**2

    def shape(site):
        try:
            x, y, z = site.pos
        except AttributeError:
            x, y, z = site
        n = (y + 1j * z) * np.exp(1j * angle)
        y, z = n.real, n.imag
        rsq = y**2 + z**2
        shape_yz = (r_in_sq <= rsq < r_out_sq and
                    z >= np.cos(coverage_angle) * np.sqrt(rsq))
        return (shape_yz and L0 <= x < L) if L > 0 else shape_yz

    r_mid = (r_out + r_in) / 2
    start_coords = np.array([L - a,
                             r_mid * np.sin(angle),
                             r_mid * np.cos(angle)])

    return shape, start_coords


def at_interface(site1, site2, shape1, shape2):
    return ((shape1[0](site1) and shape2[0](site2)) or
            (shape2[0](site1) and shape1[0](site2)))


def change_hopping_at_interface(syst, template, shape1, shape2):
    for (site1, site2), hop in syst.hopping_value_pairs():
        if at_interface(site1, site2, shape1, shape2):
            syst[site1, site2] = template[site1, site2]
    return syst


# System construction

def make_3d_wire(a, L, r1, r2, coverage_angle, angle, onsite_disorder,
                 with_leads, with_shell, shape, A_correction,
                 right_lead=True, L_barrier=None, rotate_spin_orbit=False):
    """Create a cylindrical 3D wire covered with a
    superconducting (SC) shell, but without superconductor in
    leads.

    Parameters
    ----------
    a : int
        Discretization constant in nm.
    L : int
        Length of wire (the scattering part with SC shell.)
    r1 : int
        Radius of normal part of wire in nm.
    r2 : int
        Radius of superconductor in nm.
    coverage_angle : int
        Coverage angle of superconductor in degrees.
    angle : int
        Angle of tilting of superconductor from top in degrees.
    onsite_disorder : bool
        When True, disorder in SM and requires `disorder` and `salt` aguments.
    with_leads : bool
        If True it adds infinite semiconducting leads.
    with_shell : bool
        Adds shell to the scattering area. If False no SC shell is added and
        only a cylindrical or square wire will be created.
    shape : str
        Either `circle` or `square` shaped cross section.

    Returns
    -------
    syst : kwant.builder.FiniteSystem
        The finilized kwant system.

    Examples
    --------
    This doesn't use default parameters because the variables need to be saved,
    to a file. So I create a dictionary that is passed to the function.

    >>> syst_params = dict(a=10, angle=0, site_disorder=False,
    ...                    L=30, coverage_angle=135, r1=50, r2=70,
                           shape='square', with_leads=True, with_shell=True)
    >>> syst, hopping = make_3d_wire(**syst_params)

    """
    if L_barrier is None:
        L_barrier = a
    L += L_barrier

    syst = kwant.Builder()

    if shape == 'square':
        shape_function = square_sector
    elif shape == 'circle':
        shape_function = cylinder_sector
    else:
        raise(NotImplementedError('Only square or circle wire cross'
                                  'section allowed'))

    shape_normal = shape_function(r_out=r1, angle=angle, L0=L_barrier, L=L, a=a)
    shape_barrier = shape_function(r_out=r1, angle=angle, L=L_barrier, a=a)
    shape_sc = shape_function(r_out=r2, r_in=r1, coverage_angle=coverage_angle,
                              angle=angle, L0=L_barrier, L=L, a=a)

    delta_barrier = (L_barrier == a)
    templ_sm, templ_sc, templ_interface, templ_barrier = discretized_hamiltonian(
        a, delta_barrier, False, rotate_spin_orbit)

    templ_sm = apply_peierls_to_template(templ_sm)
    templ_barrier = apply_peierls_to_template(templ_barrier)

    if onsite_disorder:
        templ_sm = add_disorder_to_template(templ_sm)

    syst.fill(templ_sm, *shape_normal)
    if L_barrier != 0:
        syst.fill(templ_barrier, *shape_barrier)

    if with_shell:

        if A_correction:
            lat = templ_sc.lattice
            xyz_offset = get_offset(*shape_sc, lat=lat)
        else:
            xyz_offset = (0, 0, 0)

        templ_sc = apply_peierls_to_template(templ_sc, xyz_offset=xyz_offset)
        syst.fill(templ_sc, *shape_sc)

        # Adding a tunnel barrier between SM and SC
        templ_interface = apply_peierls_to_template(templ_interface)
        syst = change_hopping_at_interface(syst, templ_interface,
                                           shape_normal, shape_sc)

    if with_leads:
        lead = make_lead(a, r1, r2, coverage_angle, angle, rotate_spin_orbit,
                         A_correction=False, with_shell=False, shape=shape)
        # The lead at the side of the tunnel barrier.
        syst.attach_lead(lead.reversed())

        # The second lead on the other side.
        if right_lead:
            syst.attach_lead(lead)

    return syst.finalized()


def make_lead(a, r1, r2, coverage_angle, angle, rotate_spin_orbit,
              A_correction, with_shell, shape, unit_cells=1):
    """Create an infinite cylindrical 3D wire partially covered with a
    superconducting (SC) shell.

    Parameters
    ----------
    a : int
        Discretization constant in nm.
    r1 : int
        Radius of normal part of wire in nm.
    r2 : int
        Radius of superconductor in nm.
    coverage_angle : int
        Coverage angle of superconductor in degrees.
    angle : int
        Angle of tilting of superconductor from top in degrees.
    with_shell : bool
        Adds shell to the scattering area. If False no SC shell is added and
        only a cylindrical or square wire will be created.
    shape : str
        Either `circle` or `square` shaped cross section.
    unit_cells : integer
        Number of translational unit cells to include in the lead.

    Returns
    -------
    syst : kwant.builder.InfiniteSystem
        The finilized kwant system.

    Examples
    --------
    This doesn't use default parameters because the variables need to be saved,
    to a file. So I create a dictionary that is passed to the function.

    >>> syst_params = dict(a=10, angle=0, coverage_angle=185, r1=50,
    ...                    r2=70, A_correction=True, shape='square',
    ...                    with_shell=True, rotate_spin_orbit=False)
    >>> syst, hopping = make_lead(**syst_params)

    """
    if shape == 'square':
        shape_function = square_sector
    elif shape == 'circle':
        shape_function = cylinder_sector
    else:
        raise NotImplementedError('Only square or circle wire cross section allowed')

    shape_normal_lead = shape_function(r_out=r1, angle=angle, L=-1, a=a)
    shape_sc_lead = shape_function(r_out=r2, r_in=r1, coverage_angle=coverage_angle, angle=angle, L=-1, a=a)

    sz = np.array([[1, 0], [0, -1]])
    cons_law = np.kron(np.eye(2), -sz)
    symmetry = kwant.TranslationalSymmetry((unit_cells*a, 0, 0))
    lead = kwant.Builder(symmetry, conservation_law=cons_law)

    templ_sm, templ_sc, templ_interface, _ = discretized_hamiltonian(
        a, as_lead=True, rotate_spin_orbit=rotate_spin_orbit)
    templ_sm = apply_peierls_to_template(templ_sm)
    lead.fill(templ_sm, *shape_normal_lead)

    if with_shell:
        lat = templ_sc.lattice
        # Take only a slice of SC instead of the infinite shape_sc_lead
        shape_sc = shape_function(r_out=r2, r_in=r1, coverage_angle=coverage_angle, angle=angle, L=a, a=a)

        if A_correction:
            xyz_offset = get_offset(*shape_sc, lat)
        else:
            xyz_offset = (0, 0, 0)

        templ_sc = apply_peierls_to_template(templ_sc, xyz_offset=xyz_offset)
        templ_interface = apply_peierls_to_template(templ_interface)
        lead.fill(templ_sc, *shape_sc_lead)

        # Adding a tunnel barrier between SM and SC
        lead = change_hopping_at_interface(lead, templ_interface,
                                           shape_normal_lead, shape_sc_lead)

    return lead


def make_simple_3d_wire(a, L, r, with_leads, shape, right_lead=True,
                        L_barrier=None, rotate_spin_orbit=False):
    """Create a cylindrical 3D wire with intrinsic
    superconductivity (SC), but SC in the leads.

    Parameters
    ----------
    a : int
        Discretization constant in nm.
    L : int
        Length of wire (the scattering part with SC shell.)
    r : int
        Radius of the wire in nm.
    with_leads : bool
        If True it adds infinite semiconducting leads.
    shape : str
        Either `circle` or `square` shaped cross section.

    Returns
    -------
    syst : kwant.builder.FiniteSystem
        The finilized kwant system.

    Examples
    --------
    This doesn't use default parameters because the variables need to be saved,
    to a file. So I create a dictionary that is passed to the function.

    >>> syst_params = dict(a=10, angle=0, site_disorder=False, with_leads=True,
    ...                    L=30, r=35, shape='square', right_lead=True)
    >>> syst, hopping = make_simple_3d_wire(**syst_params)

    """
    if L_barrier is None:
        L_barrier = a
    L += L_barrier

    syst = kwant.Builder()

    if shape == 'square':
        shape_function = square_sector
    elif shape == 'circle':
        shape_function = cylinder_sector
    else:
        raise(NotImplementedError('Only square or circle wire cross'
                                  'section allowed'))

    shape_sc = shape_function(r_out=r, angle=0, L0=L_barrier, L=L, a=a)
    shape_barrier = shape_function(r_out=r, angle=0, L=L_barrier, a=a)

    delta_barrier = (L_barrier == a)
    _, templ_sc, templ_barrier = discretized_hamiltonian(
        a, delta_barrier, False, rotate_spin_orbit, intrinsic_sc=True)

    templ_sc = apply_peierls_to_template(templ_sc)
    templ_barrier = apply_peierls_to_template(templ_barrier)

    syst.fill(templ_sc, *shape_sc)
    if L_barrier != 0:
        syst.fill(templ_barrier, *shape_barrier)

    if with_leads:
        lead = make_simple_lead(a, r, rotate_spin_orbit, shape,
                                superconducting=False)
        # The lead at the side of the tunnel barrier.
        syst.attach_lead(lead.reversed())

        # The second lead on the other side.
        if right_lead:
            syst.attach_lead(lead)

    return syst.finalized()


def make_simple_lead(a, r, rotate_spin_orbit, shape, superconducting):
    """Create an infinite cylindrical 3D wire partially covered with a
    superconducting (SC) shell.

    Parameters
    ----------
    a : int
        Discretization constant in nm.
    r : int
        Radius of the wire in nm.
    shape : str
        Either `circle` or `square` shaped cross section.
    superconducting : bool
        Make the lead superconducting or a normal metal.

    Returns
    -------
    syst : kwant.builder.InfiniteSystem
        The finilized kwant system.

    Examples
    --------
    This doesn't use default parameters because the variables need to be saved,
    to a file. So I create a dictionary that is passed to the function.

    >>> syst_params = dict(a=10, r=35, shape='square', rotate_spin_orbit=False,
    ...                    superconducting=True)
    >>> syst, hopping = make_simple_lead(**syst_params)

    """
    if shape == 'square':
        shape_function = square_sector
    elif shape == 'circle':
        shape_function = cylinder_sector
    else:
        raise NotImplementedError('Only square or circle wire cross section allowed')

    shape_lead = shape_function(r_out=r, angle=0, L=-1, a=a)

    sz = np.array([[1, 0], [0, -1]])
    cons_law = np.kron(np.eye(2), -sz)
    symmetry = kwant.TranslationalSymmetry((a, 0, 0))
    lead = kwant.Builder(symmetry, conservation_law=cons_law)

    templ_sm, templ_sc, _ = discretized_hamiltonian(
        a, as_lead=True, rotate_spin_orbit=rotate_spin_orbit, intrinsic_sc=True)

    if superconducting:
        templ_sc = apply_peierls_to_template(templ_sc)
        lead.fill(templ_sc, *shape_lead)
    else:
        templ_sm = apply_peierls_to_template(templ_sm)
        lead.fill(templ_sm, *shape_lead)

    return lead


def fix_simple_params(params, syst_pars, Delta=0.25):
    """Helper function to change params for `make_3d_wire`
    to `make_simple_3d_wire`."""
    params = copy(params)
    syst_pars = copy(syst_pars)
    params['Delta'] = Delta
    syst_pars['r'] = syst_pars['r1']
    for k in ['coverage_angle', 'angle', 'r1', 'r2',
              'A_correction', 'with_shell', 'onsite_disorder']:
        syst_pars.pop(k, None)
    return params, syst_pars


# Physics functions
def andreev_conductance(syst, params, E):
    """The Andreev conductance is N - R_ee + R_he."""
    smatrix = kwant.smatrix(syst, energy=E, params=params)
    r_ee = smatrix.transmission((0, 0), (0, 0))
    r_he = smatrix.transmission((0, 1), (0, 0))
    N_e = smatrix.submatrix((0, 0), (0, 0)).shape[0]
    return N_e - r_ee + r_he


def conductance(syst, params, E):
    smatrix = kwant.smatrix(syst, energy=E, params=params)
    return smatrix.transmission(0, 1)


def bands(lead, params, ks=None):
    if ks is None:
        ks = np.linspace(-3, 3)

    bands = kwant.physics.Bands(lead, params=params)

    if isinstance(ks, (float, int)):
        return bands(ks)
    else:
        return np.array([bands(k) for k in ks])


def translation_ev(h, t, tol=1e6):
    """Compute the eigenvalues of the translation operator of a lead.

    Adapted from kwant.physics.leads.modes.

    Parameters
    ----------
    h : numpy array, real or complex, shape (N, N) The unit cell
        Hamiltonian of the lead unit cell.
    t : numpy array, real or complex, shape (N, M)
        The hopping matrix from a lead cell to the one on which self-energy
        has to be calculated (and any other hopping in the same direction).
    tol : float
        Numbers and differences are considered zero when they are smaller
        than `tol` times the machine precision.

    Returns
    -------
    ev : numpy array
        Eigenvalues of the translation operator in the form lambda=r*exp(i*k),
        for |r|=1 they are propagating modes.
    """
    a, b = kwant.physics.leads.setup_linsys(h, t, tol, None).eigenproblem
    ev = kwant.physics.leads.unified_eigenproblem(a, b, tol=tol)[0]
    return ev


def cell_mats(lead, params, bias=0):
    h = lead.cell_hamiltonian(params=params)
    h -= bias * np.identity(len(h))
    t = lead.inter_cell_hopping(params=params)
    return h, t


def gap_minimizer(lead, params, energy):
    """Function that minimizes a function to find the band gap.
    This objective function checks if there are progagating modes at a
    certain energy. Returns zero if there is a propagating mode.

    Parameters
    ----------
    lead : kwant.builder.InfiniteSystem object
        The finalized infinite system.
    params : dict
        A dict that is used to store Hamiltonian parameters.
    energy : float
        Energy at which this function checks for propagating modes.

    Returns
    -------
    minimized_scalar : float
        Value that is zero when there is a propagating mode.
    """
    h, t = cell_mats(lead, params, bias=energy)
    ev = translation_ev(h, t)
    norm = (ev * ev.conj()).real
    return np.min(np.abs(norm - 1))


def find_gap(lead, params, tol=1e-6):
    """Finds the gapsize by peforming a binary search of the modes with a
    tolarance of tol.

    Parameters
    ----------
    lead : kwant.builder.InfiniteSystem object
        The finalized infinite system.
    params : dict
        A dict that is used to store Hamiltonian parameters.
    tol : float
        The precision of the binary search.

    Returns
    -------
    gap : float
        Size of the gap.
    """
    lim = [0, np.abs(bands(lead, params, ks=0)).min()]
    if gap_minimizer(lead, params, energy=0) < 1e-15:
        # No band gap
        gap = 0
    else:
        while lim[1] - lim[0] > tol:
            energy = sum(lim) / 2
            par = gap_minimizer(lead, params, energy)
            if par < 1e-10:
                lim[1] = energy
            else:
                lim[0] = energy
        gap = sum(lim) / 2
    return gap


def get_cross_section(syst, pos, direction):
    coord = np.array([s.pos for s in syst.sites if s.pos[direction] == pos])
    cross_section = np.delete(coord, direction, 1)
    return cross_section


def get_densities(lead, k, params):
    xy = get_cross_section(lead, pos=0, direction=0)
    h, t = lead.cell_hamiltonian(params=params), lead.inter_cell_hopping(params=params)
    h_k = h + t * np.exp(1j * k) + t.T.conj() * np.exp(-1j * k)

    vals, vecs = np.linalg.eigh(h_k)
    indxs = np.argsort(np.abs(vals))
    vecs = vecs[:, indxs]
    vals = vals[indxs]

    norbs = lat_from_syst(lead).norbs
    densities = np.linalg.norm(vecs.reshape(-1, norbs, len(vecs)), axis=1)**2
    return xy, vals, densities.T


def V_barrier(x, V_barrier_height, V_barrier_mu, V_barrier_sigma):
    return common.gaussian(x=x, a=V_barrier_height, mu=V_barrier_mu,
                    sigma=V_barrier_sigma)


def is_antisymmetric(H):
    return np.allclose(-H, H.T)


def get_h_k(lead, params):
    h, t = cell_mats(lead, params)
    h_k = lambda k: h + t * np.exp(1j * k) + t.T.conj() * np.exp(-1j * k)
    return h_k


def make_skew_symmetric(ham):
    """
    Makes a skew symmetric matrix by a matrix multiplication of a unitary
    matrix U. This unitary matrix is taken from the Topology MOOC 0D, but
    that is in a different basis. To get to the right basis one multiplies
    by [[np.eye(2), 0], [0, sigma_y]].

    Parameters:
    -----------
    ham : numpy.ndarray
        Hamiltonian matrix gotten from sys.cell_hamiltonian()

    Returns:
    --------
    skew_ham : numpy.ndarray
        Skew symmetrized Hamiltonian
    """
    W = ham.shape[0] // 4
    I = np.eye(2, dtype=complex)
    sigma_y = np.array([[0, 1j], [-1j, 0]], dtype=complex)
    U_1 = np.bmat([[I, I], [1j * I, -1j * I]])
    U_2 = np.bmat([[I, 0 * I], [0 * I, sigma_y]])
    U = U_1 @ U_2
    U = np.kron(np.eye(W, dtype=complex), U)
    skew_ham = U @ ham @ U.H

    assert is_antisymmetric(skew_ham)

    return skew_ham


def calculate_pfaffian(lead, params):
    """
    Calculates the Pfaffian for the infinite system by computing it at k = 0
    and k = pi.

    Parameters:
    -----------
    lead : kwant.builder.InfiniteSystem object
          The finalized system.

    """
    h_k = get_h_k(lead, params)

    skew_h0 = make_skew_symmetric(h_k(0))
    skew_h_pi = make_skew_symmetric(h_k(np.pi))

    pf_0 = np.sign(pf.pfaffian(1j * skew_h0, sign_only=True).real)
    pf_pi = np.sign(pf.pfaffian(1j * skew_h_pi, sign_only=True).real)
    pfaf = pf_0 * pf_pi

    return pfaf


def get_potential(params, syst_pars):
    """Get a potential shape for in the wire.

    Returns
    -------
    pot : function
        Potential function that takes (x, z, V_0).
    """
    # Only passing the params that are used (`_params`) for caching purposes
    _params = common.select_keys(params, ('sigma', 'V_l', 'V_r',
                                          'x0', 'V_0_top'))
    pot_params = common.get_smooth_bump_params(_params)
    V_top = common.smooth_bump(params, pot_params)
    V_bottom = partial(common.gaussian,
                       a=params['V_0'],
                       mu=params['x0'],
                       sigma=params['sigma'])
    z0 = -syst_pars['r1']
    z1 = syst_pars['r1']
    return lambda x, z, V_0: (V_bottom(x, a=V_0) * (z1 - z) + V_top(x) * (z - z0)) / (z1 - z0)


def get_potential2(params, syst_pars):
    V_0 = params['V_0']
    V_r = params['V_r']
    V_l = params['V_l']
    x0 = params['x0']
    sigma = params['sigma']
    V_bottom = lambda x, V_0: (common.gaussian(x, V_0, x0, sigma) if x > x0 else V_0)
    V_top = lambda x: (V_r + common.gaussian(x, V_l - V_r, x0, sigma) if x > x0 else V_l)
    z0 = -syst_pars['r1']
    z1 = syst_pars['r1']
    return lambda x, z, V_0: (V_bottom(x, V_0) * (z1 - z) + V_top(x) * (z - z0)) / (z1 - z0)


def get_potential2_lead(params, syst_pars):
    V_r = params['V_r']
    V_bottom = 0
    V_top = V_r
    z0 = -syst_pars['r1']
    z1 = syst_pars['r1']
    return lambda x, y, z: (V_bottom * (z1 - z) + V_top * (z - z0)) / (z1 - z0)
