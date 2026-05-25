import jax
from flax import linen as nn
from jax import numpy as jnp
from typing import Sequence, Callable, Any
from .wavefunctions import ElecConf

def build_LogFid_network(model: nn.Module, target_fn: Callable):
    """Create loss function for -log(F), assume sampling from network"""

    def eval_local(params, x: ElecConf):
        """Local loss function"""
        sign_network, log_network = model.apply(params, x)
        sign_target, log_target = target_fn(params, x)

        log_alpha = (log_target - log_network) + 1j*(jnp.angle(sign_target) - jnp.angle(sign_network))

        local_comps = {
            'log_alpha': log_alpha,
        }
        return local_comps, sign_network, log_network

    batch_local = jax.vmap(eval_local, in_axes=(None, 0), out_axes=0)

    def eval_total(params, data):
        """Batched loss function"""
        conf_net = data['network']

        local_comps, sign, logf = batch_local(params, conf_net)

        if jnp.iscomplexobj(sign):
            logf += jnp.log(sign)

        log_alpha = local_comps['log_alpha'] - jnp.mean(jnp.real(local_comps['log_alpha']))

        alpha = jnp.exp(log_alpha)
        alpha_mean = jnp.mean(alpha)

        loss_local = jax.lax.stop_gradient(1 - alpha/alpha_mean)

        loss = 2*jnp.mean(loss_local * jnp.conj(logf)).real

        local_var = jnp.mean(loss_local * jnp.conj(loss_local)).real - jnp.abs(jnp.mean(loss_local))**2

        alpha_sq = jnp.abs(alpha)**2
        alpha_sq_mean = jnp.mean(alpha_sq)
        fidelity = jnp.abs(alpha_mean)**2 / alpha_sq_mean

        aux = dict(actual_loss = fidelity, local_var = local_var, nans = jnp.isnan(loss_local).sum())
        return loss, aux
    
    return eval_total


def build_Prob_mix(model: nn.Module, target_fn: Callable):
    """Create loss function for mixed probability loss"""

    def eval_local(params, x: ElecConf):
        sign_network, log_network = model.apply(params, x)
        _, log_target = target_fn(params, x)

        eloc = 2*log_network - 2*log_target

        local_comps = {
            'ratio': eloc,
        }

        return local_comps, sign_network, log_network

    batch_local = jax.vmap(eval_local, in_axes=(None, 0), out_axes=0)

    def eval_total(params, data):
        """Batched loss function"""

        conf_net = data['network']
        conf_tgt = data['target']

        # Network samples
        local_net, sign_net, logf_net = batch_local(params, conf_net)
        local_tgt, _, _ = batch_local(params, conf_tgt)

        if jnp.iscomplexobj(sign_net):
            logf_net += jnp.log(sign_net)

        ratio_net = local_net['ratio'] 
        ratio_tgt = local_tgt['ratio']


        ratio_mix = jnp.concatenate([ratio_net, ratio_tgt], axis=0)
        ratio_mean = jnp.mean(ratio_mix)


        delta_net = (ratio_net - jax.lax.stop_gradient(ratio_mean))**2
        delta_tgt = (ratio_tgt - jax.lax.stop_gradient(ratio_mean))**2

        #Loss_network
        loss_local_net = jax.lax.stop_gradient(delta_net - jnp.mean(delta_net))
        loss_net = jnp.mean(2 * loss_local_net * logf_net.real + delta_net) #The extra term takes account of self sampling


        loss_tgt = jnp.mean(delta_tgt)

        loss = loss_net + loss_tgt

        delta_mix = jnp.concatenate([delta_net, delta_tgt], axis=0)
        delta_mean_mix = jnp.mean(delta_mix)

        local_var = (
            jnp.mean(delta_mix * jnp.conj(delta_mix)).real
            - jnp.abs(delta_mean_mix)**2
        )

        nans = jnp.isnan(
            jnp.concatenate([loss_local_net, delta_tgt], axis=0)
        ).sum()

        aux = dict(
            loss = delta_mean_mix,
            local_var   = local_var,
            nans        = nans,
        )

        return loss, aux

    return eval_total



def _select_output(f: Callable[..., Sequence[Any]],
                  argnum: int) -> Callable[..., Any]:
  """Return the argnum-th result from callable f."""

  def f_selected(*args, **kwargs):
    return f(*args, **kwargs)[argnum]

  return f_selected


def build_Current_network(model: nn.Module, target_fn: Callable):
    """Create loss function for current loss"""

    def _phase_target_fn(params, e_pos, e_spin):
        sign = _select_output(target_fn, 0)(params, (e_pos, e_spin))
        return jnp.angle(sign)

    def _phase_network_fn(params, e_pos, e_spin):
        sign = _select_output(model.apply, 0)(params, (e_pos, e_spin))
        return jnp.angle(sign)

    grad_phase_target = jax.grad(lambda params, e_pos, e_spin: _phase_target_fn(params, e_pos, e_spin), argnums=1)
    grad_phase_network = jax.grad(lambda params, e_pos, e_spin: _phase_network_fn(params, e_pos, e_spin), argnums=1)

    def grad_target_fn(params, x: ElecConf):
        e_pos, e_spin = x
        return grad_phase_target(params, e_pos, e_spin)

    def grad_network_fn(params, x: ElecConf):
        e_pos, e_spin = x
        return grad_phase_network(params, e_pos, e_spin)


    def eval_local(params, x: ElecConf):
        sign_network, log_network = model.apply(params, x)

        nabla_target = grad_target_fn(params, x)
        nabla_network = grad_network_fn(params, x)

        l2_nabla = jnp.sum(jnp.linalg.norm(nabla_network - nabla_target, axis=1) ** 2)

        local_comps = {
            'l2_nabla': l2_nabla,
        }

        return local_comps, sign_network, log_network

    batch_local = jax.vmap(eval_local, in_axes=(None, 0), out_axes=0)

    def eval_total(params, data):

        conf = data['network']
        local_comps, sign, logf = batch_local(params, conf)

        if jnp.iscomplexobj(sign):
            logf += jnp.log(sign)

        l2_nabla = local_comps['l2_nabla']

        loss_local = jax.lax.stop_gradient(l2_nabla - jnp.mean(l2_nabla))

        loss = jnp.mean(2* loss_local * logf.real + l2_nabla)

        local_var = jnp.mean(l2_nabla * jnp.conj(l2_nabla)).real - jnp.abs(jnp.mean(l2_nabla))**2

        aux = dict(loss = jnp.mean(l2_nabla), local_var = local_var, nans = jnp.isnan(loss_local).sum())

        return loss, aux

    return eval_total

LOSS_FACTORIES = {
    'LogFid_network': build_LogFid_network,
    'Prob_mix': build_Prob_mix,
    'Current_network': build_Current_network,
}



        





