from functools import cached_property

import numpy as np
import scipy.linalg as la

from .linalg import allclose, prop_to_id, solve_mat_eqn, symmetry_adapted_sun
from .groups import (
    PointGroup,
    _copy_point_group_element,
    character_table_burnside,
    conjugacy_classes,
    full_rotation,
)


class PointGroupAnalysis:
    """Derived representation-theory data for a PointGroup."""

    def __init__(self, group):
        if not isinstance(group, PointGroup):
            raise ValueError("PointGroupAnalysis requires a PointGroup.")
        self.group = group

    @cached_property
    def _fixed_group(self):
        return self.group.fix_U_phases()

    @cached_property
    def _fixed_analysis(self):
        # Character formulas assume a consistent representative U for each abstract
        # group element, so reuse the phase-fixed copy when needed.
        if self._fixed_group is self.group:
            return self
        return self._fixed_group.analysis

    @cached_property
    def _conjugacy_data(self):
        # All character-table machinery runs on the unitary subgroup; antiunitary
        # generators are folded back in only when constructing physical irreps.
        return conjugacy_classes(self.group.unitary_elements)

    @property
    def conjugacy_classes(self):
        return self._conjugacy_data[0]

    @property
    def class_representatives(self):
        return self._conjugacy_data[1]

    @property
    def class_by_element(self):
        return self._conjugacy_data[2]

    @cached_property
    def character_table(self):
        return character_table_burnside(
            self.group.unitary_elements,
            self.conjugacy_classes,
            self.class_by_element,
            tol=self.group.tol,
        )

    @cached_property
    def character_table_full(self):
        # Expand class characters to one entry per sorted unitary group element.
        return self.character_table[
            :,
            np.array(
                [self.class_by_element[g] for g in self.group.unitary_elements_list]
            ),
        ]

    @property
    def character(self):
        return np.trace([g.U for g in self.class_representatives], axis1=-1, axis2=-2)

    @property
    def character_full(self):
        return np.trace(
            [g.U for g in self.group.unitary_elements_list], axis1=-1, axis2=-2
        )

    @cached_property
    def decompose_U_rep(self):
        fixed = self._fixed_analysis
        decomp = (
            fixed.character_table_full
            @ fixed.character_full.conj()
            / len(fixed.group.unitary_elements)
        )
        decomp_round = np.around(decomp).real.astype(int)
        if not allclose(decomp, decomp_round):
            raise ValueError("Invalid characters, the product should be integer.")
        return decomp_round

    @cached_property
    def decompose_R_rep(self):
        char = np.trace(
            [g.R for g in self.group.unitary_elements_list], axis1=-1, axis2=-2
        )
        decomp = (
            self.character_table_full @ char.conj() / len(self.group.unitary_elements)
        )
        decomp_round = np.around(decomp).real.astype(int)
        if not allclose(decomp, decomp_round):
            raise ValueError("Invalid characters, the product should be integer.")
        return decomp_round

    @cached_property
    def symmetry_adapted_basis(self):
        fixed = self._fixed_analysis
        bases = []
        for chi, n in zip(fixed.character_table_full, fixed.decompose_U_rep):
            if n == 0:
                continue
            d = int(np.around(chi[0]).real)
            basis_chi = np.empty((fixed.group.U_shape[0], 0))
            for v in np.eye(fixed.group.U_shape[0]):
                w = np.sum(
                    [
                        chi[i].conj() * g.U @ v
                        for i, g in enumerate(fixed.group.unitary_elements_list)
                    ],
                    axis=0,
                )
                w *= chi[0] / len(fixed.group.unitary_elements)
                if np.linalg.norm(w) <= fixed.group.tol:
                    continue
                if n == 1 and d == 1:
                    for i, g in enumerate(fixed.group.unitary_elements_list):
                        assert allclose(chi[i] * w, g.U @ w)
                wspan = np.array([g.U @ w for g in fixed.group.unitary_elements]).T
                basis_chi = np.hstack([basis_chi, wspan])
                rank = np.linalg.matrix_rank(basis_chi, fixed.group.tol)
                assert rank <= n * d, (rank, n, d)
                if rank == n * d:
                    break
            basis_chi = la.qr(basis_chi, pivoting=True)[0]
            basis_chi = basis_chi[:, :rank]

            # Within one isotypic block, align the n equivalent copies so later
            # projections return one basis per copy instead of an arbitrary mixture.
            gens = np.array(
                [
                    basis_chi.T.conj() @ g.U @ basis_chi
                    for g in fixed.group.minimal_generators
                ]
            )
            vecs = symmetry_adapted_sun(gens, n=n)
            for i in range(n):
                bases.append(basis_chi @ vecs[:, :, i].T)
        return bases

    @cached_property
    def regular_representation(self):
        fixed = self._fixed_analysis
        new_generators = set()
        element_dict = {g: i for i, g in enumerate(fixed.group.unitary_elements)}
        for g in fixed.group.minimal_generators:
            # Left-regular action of the unitary subgroup in the basis of group elements.
            mat = np.zeros(
                (len(fixed.group.unitary_elements), len(fixed.group.unitary_elements)),
                dtype=int,
            )
            for h in fixed.group.unitary_elements:
                mat[element_dict[h], element_dict[g * h]] = 1
            new_g = _copy_point_group_element(g)
            new_g.U = mat
            new_generators.add(new_g)
        reg_rep = type(fixed.group)(
            new_generators,
            double_group=fixed.group.double_group,
            tol=fixed.group.tol,
        )
        if fixed.group._tests:
            assert reg_rep.consistent_U
            assert allclose(
                [g.R for g in fixed.class_representatives],
                [g.R for g in reg_rep.analysis.class_representatives],
            )
            assert allclose(reg_rep.analysis.character_table, fixed.character_table)
            assert allclose(
                reg_rep.analysis.decompose_U_rep, reg_rep.analysis.character_table[:, 0]
            )
        return reg_rep

    def antiunitary_conjugate_characters(self):
        TR = self.group.antiunitary_generator
        return np.array(
            [
                [
                    chi[
                        self.group.unitary_elements_list.index(TR * g * TR.inv())
                    ].conj()
                    for g in self.group.unitary_elements_list
                ]
                for chi in self.character_table_full
            ]
        )

    @cached_property
    def irreps(self):
        fixed = self._fixed_analysis
        reg_rep = fixed.regular_representation
        reg_analysis = reg_rep.analysis
        irreps = []
        bases = reg_analysis.symmetry_adapted_basis
        m = 0
        for i, n in enumerate(reg_analysis.decompose_U_rep):
            basis_chi = bases[m]
            new_generators = set()
            for g in reg_rep.minimal_generators:
                new_g = _copy_point_group_element(g)
                new_g.U = basis_chi.T.conj() @ g.U @ basis_chi
                assert allclose(
                    np.trace(new_g.U),
                    reg_analysis.character_table_full[
                        i, reg_analysis.group.unitary_elements_list.index(g)
                    ],
                )
                if fixed.group._tests:
                    assert allclose(g.U.T.conj() @ g.U, np.eye(g.U.shape[1]))
                    assert allclose(
                        new_g.U.T.conj() @ new_g.U, np.eye(new_g.U.shape[1])
                    )
                new_generators.add(new_g)
            irrep = type(fixed.group)(
                new_generators,
                double_group=fixed.group.double_group,
                tol=fixed.group.tol,
            )
            assert len(fixed.group.unitary_elements) == len(irrep.unitary_elements)
            if fixed.group._tests:
                assert (
                    irrep.analysis.class_representatives
                    == reg_analysis.class_representatives
                )
                assert irrep.consistent_U
                assert allclose(
                    irrep.analysis.character_table, reg_analysis.character_table
                )
                assert irrep.consistent_U
                assert allclose(
                    irrep.analysis.decompose_U_rep,
                    np.eye(reg_analysis.character_table.shape[0])[i],
                )
            irreps.append(irrep)
            m += n
        if fixed.group.antiunitary_generator is not None:
            irreps = fixed._physical_irreps(irreps)
        if fixed.group.force_double_group:
            irreps = [
                irr
                for irr in irreps
                if allclose(
                    full_rotation(irr.elements).U,
                    -np.eye(full_rotation(irr.elements).U.shape[0]),
                )
            ]
        return irreps

    def _physical_irreps(self, irreps):
        TR = self.group.antiunitary_generator
        chars = self.character_table_full
        # Pair unitary irreps that are related by the canonical antiunitary generator.
        conj_prod = (
            chars @ self.antiunitary_conjugate_characters().T.conj() / chars.shape[1]
        )
        conj_ind = zip(*np.nonzero(np.triu(np.around(conj_prod))))

        physical_irreps = []
        for i, j in conj_ind:
            if self.group.force_double_group and not allclose(
                full_rotation(irreps[i].elements).U,
                -np.eye(full_rotation(irreps[i].elements).U.shape[0]),
            ):
                continue

            TR2 = [g for g in irreps[i].elements if g == TR**2]
            assert len(TR2) == 1
            TR2 = TR2[0]
            irrep_dict = {g: g.U for g in irreps[i].elements}
            irrep_found = False
            if i == j:
                new_generators = set(irreps[i].minimal_generators)
                left = np.array([g.U.conj() for g in irreps[i].minimal_generators])
                right = np.array(
                    [
                        irrep_dict[TR * g * TR.inv()]
                        for g in irreps[i].minimal_generators
                    ]
                )
                TRU = solve_mat_eqn(left, right)
                assert TRU.shape[0] <= 1
                if TRU.shape[0] == 1:
                    TRU = TRU[0]
                    TRU = TRU / np.sqrt(prop_to_id(TRU @ TRU.conj().T)[1])
                    new_TR = _copy_point_group_element(TR)
                    new_TR.U = TRU
                    if allclose(TR2.U, (new_TR**2).U):
                        irrep_found = True
                        new_generators.add(new_TR)

            if not irrep_found:
                new_generators = set()
                for g in irreps[i].minimal_generators:
                    new_g = _copy_point_group_element(g)
                    new_g.U = la.block_diag(g.U, irrep_dict[TR * g * TR.inv()].conj())
                    new_generators.add(new_g)
                new_TR = _copy_point_group_element(TR)
                eye = np.eye(irreps[i].U_shape[0])
                new_TR.U = np.block([[0 * eye, TR2.U], [eye, 0 * eye]])
                new_generators.add(new_TR)

            irrep = type(self.group)(
                new_generators,
                double_group=(
                    "forced"
                    if self.group.force_double_group
                    else self.group.double_group
                ),
                tol=self.group.tol,
            )
            if self.group._tests:
                assert irrep.consistent_U
            physical_irreps.append(irrep)
        return physical_irreps

    @cached_property
    def reality(self):
        rep = self.decompose_U_rep
        if not sum(rep) == 1:
            raise ValueError("Reality is only defined for irreducible representations.")
        reality = np.sum([np.trace(g.U @ g.U) for g in self.group.unitary_elements])
        reality = reality / len(self.group.unitary_elements)
        assert abs(reality - np.around(reality)) < 1e-6
        return np.around(reality).real.astype(int)
