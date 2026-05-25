from jax import numpy as jnp
from typing import Tuple, Sequence, Union, Dict
from jax.numpy import ndarray as Array
from pfaffian import slogpf

ElecCoord = ElecSpin = Array
ElecConf = Union[ElecCoord, Tuple[ElecCoord, ElecSpin]]

def disp_min_image(displacement: Array, slattice: Array, slat_inv: Array) -> Array:
    disp_frac = displacement @ slat_inv        
    disp_frac -= jnp.round(disp_frac)   
    return disp_frac @ slattice   


AVAIL_SYS_LIST = [
   'laughlin', 'wignercrystal', 'bcs', 'mooreread',
]


def build_target_fn(name: str, nspins: Sequence[int], slattice: Array, system_kwargs: Dict):

    if name.lower() == 'laughlin':
        return make_laughlin(nspins=nspins,**system_kwargs)
    elif name.lower() == 'wignercrystal':
        return make_wignercrystal(nspins=nspins, slattice=slattice,**system_kwargs)
    elif name.lower() == 'bcs':
        return make_bcs(nspins=nspins, slattice=slattice, **system_kwargs)
    elif name.lower() == 'mooreread':
        return make_mooreread(nspins=nspins,**system_kwargs)
    else:
        raise NotImplementedError(f'{name} not implemented')

def make_laughlin(
    nspins: Sequence[int],
    m: int = 3,
    lB: float = 1.0,
    num_hole: int=0,
):
  nup, ndn = nspins[0], nspins[1]
  ne = nup + ndn
  if ndn != 0 :
    raise ValueError(f'Laughlin assumes no down-spin electrons, but got {ndn}')
  if m % 2 == 0 :
    raise ValueError(f'Laughlin ansatz assumes even m, but got {m}')
  
  def holes_on_ring(k, r):
      return jnp.array([r*jnp.exp(1j*2*jnp.pi*n/k) for n in range(k)])
  
  disk_size = jnp.sqrt(2*m*ne)*lB
  hole_pos_list = [
      jnp.array([0+0j]),                 # 1
      holes_on_ring(2, 0.5*disk_size),  # 2
      holes_on_ring(3, 0.5*disk_size),  # 3
      holes_on_ring(4, 0.5*disk_size),  # 4
      holes_on_ring(5, 0.5*disk_size),  # 5
  ]
  hole_pos = hole_pos_list[num_hole-1]/lB if num_hole !=0 else None 
  
  
  def laughlin(_, x: ElecConf):

    e_pos, _ = x

    z = (e_pos[:,0] + 1j * e_pos[:,1]) / lB
    log_abs_gauss = -0.25 * jnp.sum(jnp.abs(z)**2)

    z_pairs = z[:, None] - z[None, :]
    z_pairs = z_pairs[jnp.triu_indices_from(z_pairs, 1)]
    
    sum_log_abs_pairs = jnp.sum(jnp.log(jnp.abs(z_pairs)))
    sum_arg_pairs = jnp.sum(jnp.angle(z_pairs))

    if num_hole !=0:
      hole_factor = z[:,None] - hole_pos[None, :] 
      log_abs_hole = jnp.sum(jnp.log(jnp.abs(hole_factor)))
      sum_arg_hole = jnp.sum(jnp.angle(hole_factor))
    else:
      log_abs_hole = 0.0
      sum_arg_hole = 0.0
                  
    log_abs = m * sum_log_abs_pairs + log_abs_gauss + log_abs_hole
    phase   = m * sum_arg_pairs + sum_arg_hole

    sign = jnp.exp(1j * phase)
  
    return sign, log_abs 
  
  return laughlin


def make_mooreread(
    nspins: Sequence[int],
    m: int = 2,
    lB: float = 1.0,
    num_hole: int = 0,
):
  nup, ndn = nspins[0], nspins[1]
  ne = nup + ndn
  if ndn != 0:
    raise ValueError(f'Moore-Read ansatz assumes no down-spin electrons, but got {ndn}')
  if ne % 2 != 0:
    raise ValueError(f'Moore-Read ansatz only accepts even number of electrons, but got {ne}')
  if m % 2 != 0:
    raise ValueError(f'Moore-Read ansatz only accepts even m, but got {m}')

  # --- same hole placement convention as Laughlin ---
  def holes_on_ring(k, r):
    ang = 2 * jnp.pi * jnp.arange(k) / k
    return r * jnp.exp(1j * ang)  # (k,)

  disk_size = jnp.sqrt(2 * m * ne) * lB
  hole_pos_list = [
      jnp.array([0 + 0j]),                 # 1
      holes_on_ring(2, 0.5 * disk_size),   # 2
      holes_on_ring(3, 0.5 * disk_size),   # 3
      holes_on_ring(4, 0.5 * disk_size),   # 4
      holes_on_ring(5, 0.5 * disk_size),   # 5
  ]
  hole_pos = hole_pos_list[num_hole - 1] / lB if num_hole != 0 else None  # dimensionless

  def mooreread(_, x: ElecConf):
    e_pos, _ = x  # (ne, sdim)

    z = (e_pos[:, 0] + 1j * e_pos[:, 1]) / lB
    log_abs_gauss = -0.25 * jnp.sum(jnp.abs(z) ** 2)

    z_pairs = z[:, None] - z[None, :]
    z_pairs_triu = z_pairs[jnp.triu_indices(ne, 1)]

    sum_log_abs_pairs = jnp.sum(jnp.log(jnp.abs(z_pairs_triu) + 1e-300))
    sum_arg_pairs = jnp.sum(jnp.angle(z_pairs_triu))

    log_abs = m * sum_log_abs_pairs + log_abs_gauss
    phase   = m * sum_arg_pairs

    eye = jnp.eye(ne, dtype=z.dtype)
    den = z_pairs + eye                 
    z_inv = (1.0 / den) * (1.0 - eye)  

    sign_pfa, logabs_pfa = slogpf(z_inv)
    phase_pfa = jnp.angle(sign_pfa)

    log_abs += logabs_pfa
    phase   += phase_pfa

    if num_hole !=0:
      hole_factor = z[:,None] - hole_pos[None, :] 
      log_abs_hole = jnp.sum(jnp.log(jnp.abs(hole_factor)))
      sum_arg_hole = jnp.sum(jnp.angle(hole_factor))
      log_abs += log_abs_hole
      phase   += sum_arg_hole

    sign = jnp.exp(1j * phase)
    return sign, log_abs 

  return mooreread



def make_bcs(
    nspins,
    slattice,
    k_cut=3,
    delta_0=1.0,
    wave_type='s',
    spin_type='sgl',
):

    def _make_FForbs(pair_pos, pair_spin):
        # pair_pos: (Npair, 2)
        # pair_spin: (Npair, 2)
        FF_mat = jnp.exp(1j * (k_pts @ pair_pos.T))   # (Nk, Npair)
        fr = (gkpts[:, None] * FF_mat).sum(axis=0)    # (Npair,)
        spin_coeff = spin_fn(pair_spin)
        return fr * spin_coeff

    def _make_pts():
        # reciprocal lattice vectors; slattice rows are real-space vectors
        srec_vec = 2 * jnp.pi * jnp.linalg.inv(slattice.T)
        sb1, sb2 = srec_vec[0], srec_vec[1]

        # build a candidate rectangular grid first
        grid_line = jnp.arange(-100, 100 + 1)
        grid_x, grid_y = jnp.meshgrid(grid_line, grid_line, indexing="ij")
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)

        k_all = grid_x[:, None] * sb1[None, :] + grid_y[:, None] * sb2[None, :]
        k_norm = jnp.linalg.norm(k_all, axis=1)

        # keep every k-point inside k_cut*kF
        mask = k_norm <= (k_cut * kf + 1e-5)
        k_pts = k_all[mask]

        delta_k = gap_fn(k_pts)
        k_norm = jnp.linalg.norm(k_pts, axis=1)
        eps_k = 0.5 * (k_norm ** 2 - kf ** 2)
        E_k = jnp.sqrt(eps_k ** 2 + jnp.abs(delta_k) ** 2)

        gkpts = delta_k / (eps_k + E_k + 1e-30)
        return k_pts, gkpts

    def _gap_s(k_vecs):
        return delta_0 * jnp.ones_like(k_vecs[:, 0])

    def _gap_px(k_vecs):
        return delta_0 * jnp.sin(k_vecs[:, 0])

    def _gap_py(k_vecs):
        return delta_0 * jnp.sin(k_vecs[:, 1])

    def _gap_pp(k_vecs):
        return delta_0 * (jnp.sin(k_vecs[:, 0]) + 1j * jnp.sin(k_vecs[:, 1]))

    def _gap_pm(k_vecs):
        return delta_0 * (jnp.sin(k_vecs[:, 0]) - 1j * jnp.sin(k_vecs[:, 1]))

    def _gap_dxy(k_vecs):
        return delta_0 * (jnp.sin(k_vecs[:, 0]) * jnp.sin(k_vecs[:, 1]))

    def _gap_dx2y2(k_vecs):
        return delta_0 * (jnp.cos(k_vecs[:, 0]) - jnp.cos(k_vecs[:, 1]))

    def _gap_fp(k_vecs):
        return delta_0 * (jnp.sin(k_vecs[:, 0]) + 1j * jnp.sin(k_vecs[:, 1])) ** 3

    def _gap_fm(k_vecs):
        return delta_0 * (jnp.sin(k_vecs[:, 0]) - 1j * jnp.sin(k_vecs[:, 1])) ** 3

    def _gap_fxx2y2(k_vecs):
        return delta_0 * (jnp.sin(k_vecs[:, 0]) * (jnp.cos(k_vecs[:, 0]) - jnp.cos(k_vecs[:, 1])))

    def _sgl(spins):
        return (spins[:, 0] - spins[:, 1]) / (2 * jnp.sqrt(2))

    def _tpl0(spins):
        return (1 - spins[:, 0] * spins[:, 1]) / (2 * jnp.sqrt(2))

    def _tplp(spins):
        return (1 + spins[:, 0]) * (1 + spins[:, 1]) / 4

    def _tplm(spins):
        return (1 - spins[:, 0]) * (1 - spins[:, 1]) / 4

    gap_dispatch = {
        's': _gap_s,
        'p_x': _gap_px,
        'p_y': _gap_py,
        'p_p': _gap_pp,
        'p_m': _gap_pm,
        'd_xy': _gap_dxy,
        'd_x2y2': _gap_dx2y2,
        'f_p': _gap_fp,
        'f_m': _gap_fm,
        'f_xx2y2': _gap_fxx2y2,
    }

    spin_dispatch = {
        'sgl': _sgl,
        'tpl-0': _tpl0,
        'tpl-p': _tplp,
        'tpl-m': _tplm,
    }

    nup, ndn = nspins
    ne = nup + ndn
    npair = ne // 2

    if ne % 2 != 0:
        raise ValueError(f'Number of electrons in BCS wavefunction must be even, but got {ne}.')

    if wave_type not in gap_dispatch:
        raise ValueError(f'Input pairing function {wave_type} not implemented')

    if spin_type in ['sgl', 'tpl-0'] and nup != ndn:
        raise ValueError(f'For spin_type {spin_type}, nup must equal ndn, but got nup={nup}, ndn={ndn}')
    elif spin_type == 'tpl-p' and ndn != 0:
        raise ValueError(f'For spin_type {spin_type}, ndn must be 0, but got {ndn}')
    elif spin_type == 'tpl-m' and nup != 0:
        raise ValueError(f'For spin_type {spin_type}, nup must be 0, but got {nup}')
    elif spin_type not in spin_dispatch:
        raise ValueError(f'Input spin type {spin_type} not implemented')

    if wave_type in ['s', 'd_xy', 'd_x2y2'] and spin_type != 'sgl':
        raise ValueError(f'wave type {wave_type} only supports singlet spin_type, but got {spin_type}')

    gap_fn = gap_dispatch[wave_type]
    spin_fn = spin_dispatch[spin_type]

    # 2D spinful free Fermi gas relation:
    # n = ne / Area, kF^2 = 2*pi*n
    area = jnp.abs(jnp.linalg.det(slattice))
    kf = jnp.sqrt(2 * jnp.pi * ne / area)

    k_pts, gkpts = _make_pts()

    print(f'BCS wavefunction: kf={kf}, area={area}, ne={ne}, num k-points={k_pts.shape[0]}', flush=True)

    def _take_det(_, x):

        e_pos, e_spin = x
        e1_pos = e_pos[:npair, :]
        e2_pos = e_pos[npair:, :]
        e1_spin = e_spin[:npair]
        e2_spin = e_spin[npair:]

        pair_pos = e1_pos[:, None, :] - e2_pos[None, :, :]
        pair_spin = jnp.stack([
            jnp.broadcast_to(e1_spin[:, None], (npair, npair)),
            jnp.broadcast_to(e2_spin[None, :], (npair, npair)),
        ], axis=-1)

        pair_pos = pair_pos.reshape(npair ** 2, 2)
        pair_spin = pair_spin.reshape(npair ** 2, 2)

        orbitals = _make_FForbs(pair_pos, pair_spin).reshape(npair, npair)
        orbitals = orbitals + 1e-30 * jnp.eye(npair, dtype=orbitals.dtype)

        sign, log_abs = jnp.linalg.slogdet(orbitals)
        return sign, log_abs

    def _take_pfa(_, x):
        e_pos, e_spin = x

        pair_pos = e_pos[:, None, :] - e_pos[None, :, :]
        pair_spin = jnp.stack([
            jnp.broadcast_to(e_spin[:, None], (ne, ne)),
            jnp.broadcast_to(e_spin[None, :], (ne, ne)),
        ], axis=-1)

        pair_pos = pair_pos.reshape(ne ** 2, 2)
        pair_spin = pair_spin.reshape(ne ** 2, 2)

        orbitals = _make_FForbs(pair_pos, pair_spin).reshape(ne, ne)
        orbitals = orbitals + 1e-30 * jnp.eye(ne, dtype=orbitals.dtype)

        sign, log_abs = slogpf(orbitals)
        return sign, log_abs

    if wave_type in ['s', 'd_xy', 'd_x2y2']:
        return _take_det
    else:
        return _take_pfa


def make_wignercrystal(
    nspins: Sequence[int],
    slattice: Array,
    ncell: Sequence[int] = (2,2),
    lattice_type: str = 'rec',
    width: float = 0.5, #Width of gaussian
    shape: str = 'gauss',
):

  nx, ny = ncell
  ne = nx * ny

  #first is spin up, second is spin down
  if lattice_type == 'v1_A':
    ne = nx*ny
    assert nspins[1] == 0 and nspins[0]==ne, 'v1_full only supports fully polarized case'
    site_coord = (jnp.array([[0,0]]), None)
  elif lattice_type == 'v2_AB':
    ne = 2*nx*ny
    assert nspins[1] == 0 and nspins[0]==ne, 'v2_full only supports fully polarized case'
    site_coord = (jnp.array([[0.0, 0.0], [1/3 /nx, 1/3 /ny]]), None)
  elif lattice_type == 'v2_AC':
    ne = 2*nx*ny
    assert nspins[1] == 0 and nspins[0]==ne, 'v2_full only supports fully polarized case'
    site_coord = (jnp.array([[0.0, 0.0], [1/2 /nx, 1/2 /ny]]), None)
  elif lattice_type == 'v2_BC':
    ne = 2*nx*ny
    assert nspins[1] == 0 and nspins[0]==ne, 'v2_full only supports fully polarized case'
    site_coord = (jnp.array([[1/3 /nx, 1/3 /ny], [1/2 /nx, 1/2 /ny]]), None)
  elif lattice_type == "v3_ABC":
    ne = 3*nx*ny
    assert nspins[1] == 0 and nspins[0]==ne, 'v3_full only supports fully polarized case'
    site_coord = (jnp.array([[0.0, 0.0],
                            [0.5/nx, 0.5/ny],
                            [1/3 /nx, 1/3 /ny]]), None)
  else:
    raise NotImplementedError(f'{lattice_type} not implemented')

  slat_inv = jnp.linalg.inv(slattice)

  # fractional steps for nx-by-ny WC cells inside the supercell
  df1 = jnp.array([1.0/nx, 0.0])
  df2 = jnp.array([0.0, 1.0/ny])

  ii, jj = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing='ij')
  cells = jnp.stack([ii.reshape(-1), jj.reshape(-1)], axis=1)  # (nx*ny,2)
  base_frac = cells[:, 0:1] * df1[None, :] + cells[:, 1:2] * df2[None, :]  # (nx*ny,2)

  site_coord_up, site_coord_dn = site_coord  # tuple

  # allow None / empty for a spin sector
  if site_coord_up is None:
      lat_pts_up = jnp.zeros((0, 2), dtype=slattice.dtype)
  else:
      lat_frac_up = (base_frac[:, None, :] + site_coord_up[None, :, :]).reshape(-1, 2)
      lat_frac_up = lat_frac_up - jnp.floor(lat_frac_up)
      lat_pts_up = lat_frac_up @ slattice  # (N_up,2)

  if site_coord_dn is None:
      lat_pts_dn = jnp.zeros((0, 2), dtype=slattice.dtype)
  else:
      lat_frac_dn = (base_frac[:, None, :] + site_coord_dn[None, :, :]).reshape(-1, 2)
      lat_frac_dn = lat_frac_dn - jnp.floor(lat_frac_dn)
      lat_pts_dn = lat_frac_dn @ slattice  # (N_dn,2)

  #jax.debug.print('lat_pts = {}', lat_pts)
  def _gauss_orbital(e_pos: Array):
    disp_up = disp_min_image(e_pos[:nspins[0], None, :] - lat_pts_up[None, :, :], slattice, slat_inv)
    disp_dn = disp_min_image(e_pos[nspins[0]:, None, :] - lat_pts_dn[None, :, :], slattice, slat_inv)
    dist_up = jnp.linalg.norm(disp_up,axis=-1)
    dist_dn = jnp.linalg.norm(disp_dn,axis=-1)

    orbitals = [jnp.exp(-width*(dist_up**2)), jnp.exp(-width*((dist_dn)**2))]

    return orbitals

  def _squeezed_gauss_orbital(e_pos: Array, squeeze: float = 0.5):

      disp_up = disp_min_image(e_pos[:nspins[0], None, :] - lat_pts_up[None, :, :], slattice, slat_inv)
      disp_dn = disp_min_image(e_pos[nspins[0]:, None, :] - lat_pts_dn[None, :, :], slattice, slat_inv)

      dx_up, dy_up = disp_up[..., 0], disp_up[..., 1]
      dx_dn, dy_dn = disp_dn[..., 0], disp_dn[..., 1]

      w0 = 1.0 / jnp.sqrt(width + 1e-12)
      wx = w0 * squeeze
      wy = w0 / squeeze

      q_up = (dx_up / wx) ** 2 + (dy_up / wy) ** 2
      q_dn = (dx_dn / wx) ** 2 + (dy_dn / wy) ** 2

      orb_up = jnp.exp(-q_up)
      orb_dn = jnp.exp(-q_dn)
      return [orb_up, orb_dn]


  def _moire_orbital(e_pos, phi: float = 0.0):
      disp_up = disp_min_image(e_pos[:nspins[0], None, :] - lat_pts_up[None, :, :], slattice, slat_inv)
      disp_dn = disp_min_image(e_pos[nspins[0]:, None, :] - lat_pts_dn[None, :, :], slattice, slat_inv)

      dx_up, dy_up = disp_up[...,0], disp_up[...,1]
      dx_dn, dy_dn = disp_dn[...,0], disp_dn[...,1]

      c = jnp.cos(phi)
      s = jnp.sin(phi)
      pi = jnp.pi
      rt3 = jnp.sqrt(3.0)

      def _V_series(dx, dy):
          r = jnp.sqrt(dx*dx + dy*dy)
          th = jnp.arctan2(dy, dx)
          sin3 = jnp.sin(3.0*th)
          cos6 = jnp.cos(6.0*th)
          return (
              -6.0 * c
              + (8.0 * pi**2) * c * r**2
              + (16.0 * pi**3 / (3.0 * rt3)) * s * sin3 * r**3
              - (8.0 * pi**4 / 3.0) * c * r**4
              - (16.0 * pi**5 / (9.0 * rt3)) * s * sin3 * r**5
              + (16.0 * pi**6 / 405.0) * c * (10.0 - cos6) * r**6
          )

      V_up = _V_series(dx_up, dy_up)
      V_dn = _V_series(dx_dn, dy_dn)
      V0_at0 = -6.0 * c

      pref = width / (8.0 * pi**2 * (c + 1e-7))  
      orb_up = jnp.exp(-pref * (V_up - V0_at0))
      orb_dn = jnp.exp(-pref * (V_dn - V0_at0))
      return [orb_up, orb_dn]




  orbital_dispatch = {
      'gauss': _gauss_orbital,
      'moire': _moire_orbital,
      'squeeze_gauss': _squeezed_gauss_orbital,
  }
  make_orbital = orbital_dispatch[shape]
  
  def wignercrystal(_, x:ElecConf):

    e_pos, _ = x

    orbitals = make_orbital(e_pos)
    
    sign_up, log_abs_up = jnp.linalg.slogdet(orbitals[0])
    sign_dn, log_abs_dn = jnp.linalg.slogdet(orbitals[1])

    sign = sign_up * sign_dn
    log_abs = log_abs_up + log_abs_dn
    
    return sign, log_abs
  
  return wignercrystal





