# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np

from pyfr.mpiutil import get_comm_rank_root, get_mpi
from pyfr.plugins.base import BasePlugin, init_csv


class InletForcingPlugin(BasePlugin):
	name = 'inletforcing'    
	systems = ['ac-euler', 'ac-navier-stokes', 'euler', 'navier-stokes']
	formulations = ['dual', 'std']

	def __init__(self, intg, cfgsect, suffix):
		super().__init__(intg, cfgsect, suffix)

		comm, rank, root = get_comm_rank_root()

		# Constant variables
		self._constants = self.cfg.items_as('constants', float)

		# Underlying elements class
		self.elementscls = intg.system.elementscls

		inletname = self.cfg.getliteral(cfgsect, 'inletname')
		self.area = self.cfg.getfloat(cfgsect, 'area')
		#self.Ain = self.cfg.getfloat(cfgsect, 'areain')
		#self.Aout = self.cfg.getfloat(cfgsect, 'areaout')
		self.mdotstar = self.cfg.getfloat(cfgsect, 'mdotstar') # Desired mass flow rate per area at inlet

		# Initialize rhou forcing
		intg.system.rhouforce = 0.0
		intg.system.mdotold = self.mdotstar

		# Boundary to integrate over
		bcin = 'bcon_{0}_p{1}'.format(inletname, intg.rallocs.prank)
		bcout = 'bcon_{0}_p{1}'.format(outletname, intg.rallocs.prank)

		# Get the mesh and elements
		mesh, elemap = intg.system.mesh, intg.system.ele_map


		# Interpolation matrices and quadrature weights
		self._m0in = m0in = {}
		self._qwtsin = qwtsin = defaultdict(list)

		self._m0out = m0out = {}
		self._qwtsout = qwtsout = defaultdict(list)

		# If we have the boundary then process the interface
		if bcin in mesh:
			# Element indices and associated face normals
			eidxs = defaultdict(list)
			norms = defaultdict(list)

			for etype, eidx, fidx, flags in mesh[bcin].astype('U4,i4,i1,i1'):
				eles = elemap[etype]

				if (etype, fidx) not in m0in:
					facefpts = eles.basis.facefpts[fidx]

					m0in[etype, fidx] = eles.basis.m0[facefpts]
					qwtsin[etype, fidx] = eles.basis.fpts_wts[facefpts]

				# Unit physical normals and their magnitudes (including |J|)
				npn = eles.get_norm_pnorms(eidx, fidx)
				mpn = eles.get_mag_pnorms(eidx, fidx)

				eidxs[etype, fidx].append(eidx)
				norms[etype, fidx].append(mpn[:, None]*npn)

			self._eidxsin = {k: np.array(v) for k, v in eidxs.items()}
			self._normsin = {k: np.array(v) for k, v in norms.items()}

	
		if bcout in mesh:
			# Element indices and associated face normals
			eidxs = defaultdict(list)
			norms = defaultdict(list)

			for etype, eidx, fidx, flags in mesh[bcout].astype('U4,i4,i1,i1'):
				eles = elemap[etype]

				if (etype, fidx) not in m0out:
					facefpts = eles.basis.facefpts[fidx]

					m0out[etype, fidx] = eles.basis.m0[facefpts]
					qwtsout[etype, fidx] = eles.basis.fpts_wts[facefpts]

				# Unit physical normals and their magnitudes (including |J|)
				npn = eles.get_norm_pnorms(eidx, fidx)
				mpn = eles.get_mag_pnorms(eidx, fidx)

				eidxs[etype, fidx].append(eidx)
				norms[etype, fidx].append(mpn[:, None]*npn)

			self._eidxsout = {k: np.array(v) for k, v in eidxs.items()}
			self._normsout = {k: np.array(v) for k, v in norms.items()}

	def __call__(self, intg):
		# MPI info
		comm, rank, root = get_comm_rank_root()

		# Solution matrices indexed by element type
		solns = dict(zip(intg.system.ele_types, intg.soln))
		ndims, nvars = self.ndims, self.nvars

		# Force vector
		rhou_in = np.zeros(ndims)
		rhou_out = np.zeros(ndims)

		for etype, fidx in self._m0in:
			# Get the interpolation operator
			m0 = self._m0in[etype, fidx]
			nfpts, nupts = m0.shape

			# Extract the relevant elements from the solution
			uupts = solns[etype][..., self._eidxsin[etype, fidx]]

			# Interpolate to the face
			ufpts = np.dot(m0, uupts.reshape(nupts, -1))
			ufpts = ufpts.reshape(nfpts, nvars, -1)
			ufpts = ufpts.swapaxes(0, 1)

			# Compute the U-momentum
			ruidx = 1
			ru = self.elementscls.con_to_pri(ufpts, self.cfg)[ruidx]

			# Get the quadrature weights and normal vectors
			qwts = self._qwtsin[etype, fidx]
			norms = self._normsin[etype, fidx]

			# Do the quadrature
			rhou_in[:ndims] += np.einsum('i...,ij,jik', qwts, ru, norms)

		for etype, fidx in self._m0out:
			# Get the interpolation operator
			m0 = self._m0out[etype, fidx]
			nfpts, nupts = m0.shape

			# Extract the relevant elements from the solution
			uupts = solns[etype][..., self._eidxsout[etype, fidx]]

			# Interpolate to the face
			ufpts = np.dot(m0, uupts.reshape(nupts, -1))
			ufpts = ufpts.reshape(nfpts, nvars, -1)
			ufpts = ufpts.swapaxes(0, 1)

			# Compute the pressure
			ruidx = 1
			ru = self.elementscls.con_to_pri(ufpts, self.cfg)[ruidx]

			# Get the quadrature weights and normal vectors
			qwts = self._qwtsout[etype, fidx]
			norms = self._normsout[etype, fidx]

			# Do the quadrature
			rhou_out[:ndims] += np.einsum('i...,ij,jik', qwts, ru, norms)


		# Current mass flow rate per area
		mdot = -rhou_in[0]/self.Area # Negative since rhou_in normal points outwards

		# Body forcing term added to maintain constant mass inflow
		ruf = intg.system.rhouforce + (1.0/intg._dt)*(self.mdotstar - 2.*mdot + intg.system.mdotold)

		# Broadcast to all ranks
		intg.system.rhouforce = float(comm.bcast(ruf, root=root))
		intg.system.mdotold = float(comm.bcast(mdot, root=root))

