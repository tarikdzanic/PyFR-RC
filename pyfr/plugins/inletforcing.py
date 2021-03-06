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
		self.mdotstar = self.cfg.getfloat(cfgsect, 'mdotstar') # Desired mass flow rate per area at inlet
		self.mdot = 0.0

		self.hasinlet = False

		# Initialize rhou forcing
		intg.system.rhouforce = 0.0
		intg.system.mdot = 0.0
		intg.system.mdotold = self.mdotstar

		# Boundary to integrate over
		bc = 'pcon_{0}_p{1}'.format(inletname, intg.rallocs.prank)

		# Get the mesh and elements
		mesh, elemap = intg.system.mesh, intg.system.ele_map


		# Interpolation matrices and quadrature weights
		self._m0 = m0 = {}
		self._qwts = qwts = defaultdict(list)

		# If we have the boundary then process the interface
		if bc in mesh:
			self.hasinlet = True
			# Element indices and associated face normals
			eidxs = defaultdict(list)
			norms = defaultdict(list)

			for etype, eidx, fidx, flags in mesh[bc].astype('U4,i4,i1,i1'):
				eles = elemap[etype]

				if (etype, fidx) not in m0:
					facefpts = eles.basis.facefpts[fidx]

					m0[etype, fidx] = eles.basis.m0[facefpts]
					qwts[etype, fidx] = eles.basis.fpts_wts[facefpts]

				# Unit physical normals and their magnitudes (including |J|)
				npn = eles.get_norm_pnorms(eidx, fidx)
				mpn = eles.get_mag_pnorms(eidx, fidx)

				eidxs[etype, fidx].append(eidx)
				norms[etype, fidx].append(mpn[:, None]*npn)

			self._eidxs = {k: np.array(v) for k, v in eidxs.items()}
			self._norms = {k: np.array(v) for k, v in norms.items()}


	def __call__(self, intg):
		# MPI info
		comm, rank, root = get_comm_rank_root()

		if not self.hasinlet:
			intg.system.mdot = comm.allreduce(0.0, op=get_mpi('sum')) 

		else:
			# Solution matrices indexed by element type
			solns = dict(zip(intg.system.ele_types, intg.soln))
			ndims, nvars = self.ndims, self.nvars

			# Force vector
			rhou = np.zeros(ndims)
			area = np.zeros(ndims)

			for etype, fidx in self._m0:
				# Get the interpolation operator
				m0 = self._m0[etype, fidx]
				nfpts, nupts = m0.shape

				# Extract the relevant elements from the solution
				uupts = solns[etype][..., self._eidxs[etype, fidx]]

				# Interpolate to the face
				ufpts = np.dot(m0, uupts.reshape(nupts, -1))
				ufpts = ufpts.reshape(nfpts, nvars, -1)
				ufpts = ufpts.swapaxes(0, 1)

				# Compute the U-momentum
				ruidx = 1
				ru = ufpts[ruidx]
				ones = np.ones(np.shape(ru))

				# Get the quadrature weights and normal vectors
				qwts = self._qwts[etype, fidx]
				norms = self._norms[etype, fidx]

				# Do the quadrature
				rhou[:ndims] += np.einsum('i...,ij,jik', qwts, ru, norms)
				area[:ndims] += -np.einsum('i...,ij,jik', qwts, ones, norms)


			self.mdot = abs((area[0]/self.area)*rhou[0]/area[0]) # Negative since rhou_in normal points outwards, normalized by portion of total inlet area

			intg.system.mdot = comm.allreduce(self.mdot, op=get_mpi('sum')) 

		if rank == root:
			# Body forcing term added to maintain constant mass inflow  -> weight by portion of total area for parallel runs
			ruf = intg.system.rhouforce + (1.0/intg._dt)*(self.mdotstar - 2.*intg.system.mdot + intg.system.mdotold)

			if (intg.system.mdot/self.mdotstar > 1.1 or intg.system.mdot/self.mdotstar < 0.9):
				print('Mass flow rate exceeds 10%% error: ', intg.system.mdot/self.mdotstar)

			# Broadcast to all ranks
			intg.system.rhouforce = float(comm.bcast(ruf, root=root))
			intg.system.mdotold = float(comm.bcast(intg.system.mdot, root=root))

		else:
			intg.system.rhouforce = float(comm.bcast(None, root=root))
			intg.system.mdotold = float(comm.bcast(None, root=root))
