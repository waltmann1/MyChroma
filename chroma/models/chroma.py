# Copyright Generate Biomedicines, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Joint model for protein complexes with applications to unconditional and conditional
protein design in a programmable manner.
"""

import copy
import inspect
from collections import defaultdict, namedtuple
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

from chroma.constants import AA20_3
from chroma.data.protein import Protein
from chroma.layers.structure.backbone import ProteinBackbone
from chroma.layers import graph
import gc
from time import time
from chroma.models import graph_backbone, graph_design


class Chroma(nn.Module):
    """Chroma: A generative model for protein design.

    Chroma is a generative model for proteins and protein complexes. It combines
    a diffusion model for generating protein backbones together with discrete
    generative models for sequence and sidechain conformations given structure.
    It enables programmatic design of proteins through a conditioning
    framework. This class provides an interface to:
        * Load model weights
        * Sample protein complexes, both unconditionally and conditionally
        * Perform sequence design of sampled backbones

    Args:
        weights_backbone (str, optional): The name of the pre-trained weights
            to use for the backbone network.

        weights_design (str, optional): The name of the pre-trained weights
            to use for the autoregressive design network.

        device (Optional[str]): The device on which to load the networks. If
            not specified, will automatically use a CUDA device if available,
            otherwise CPU.

        strict (bool): Whether to strictly enforce that all keys in `weights`
            match the keys in the model's state_dict.

        verbose (bool, optional): Show outputs from download and loading.
            Default False.
    """

    def __init__(
        self,
        weights_backbone: str = "named:public",
        weights_design: str = "named:public",
        device: Optional[str] = None,
        strict: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        import warnings

        warnings.filterwarnings("ignore")

        # If no device is explicity specified automatically set device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda" 
            else:
                device = "cpu"
        #print("device: ", device)
        #print("weights_backbone: ",weights_backbone )
        self.backbone_network = graph_backbone.load_model(
            weights_backbone, device=device, strict=strict, verbose=verbose
        ).eval()
        #print("loaded backbone")
        self.design_network = graph_design.load_model(
            weights_design,
            device=device,
            strict=strict,
            verbose=False,
        ).eval()
        #print("loaded design")

    def sample(
        self,
        # Backbone Args
        samples: int = 1,
        steps: int = 500,
        chain_lengths: List[int] = [100],
        tspan: List[float] = (1.0, 0.001),
        protein_init: Protein = None,
        conditioner: Optional[nn.Module] = None,
        langevin_factor: float = 2,
        langevin_isothermal: bool = False,
        inverse_temperature: float = 10,
        initialize_noise: bool = True,
        integrate_func: Literal["euler_maruyama", "heun"] = "euler_maruyama",
        sde_func: Literal["langevin", "reverse_sde", "ode"] = "reverse_sde",
        trajectory_length: int = 200,
        full_output: bool = False,
        # Sidechain Args
        design_ban_S: Optional[List[str]] = None,
        design_method: Literal["potts", "autoregressive"] = "potts",
        design_selection: Optional[Union[str, torch.Tensor]] = None,
        design_t: Optional[float] = 0.5,
        temperature_S: float = 0.01,
        temperature_chi: float = 1e-3,
        top_p_S: Optional[float] = None,
        regularization: Optional[str] = "LCP",
        potts_mcmc_depth: int = 500,
        potts_proposal: Literal["dlmc", "chromatic"] = "dlmc",
        potts_symmetry_order: int = None,
        verbose: bool = False,
    ) -> Union[
        Union[Protein, List[Protein]], Tuple[Union[Protein, List[Protein]], dict]
    ]:
        """
        Performs Backbone Sampling and Sequence Design and returns a Protein or list
        of Proteins. Optionally this method can return additional arguments to show
        details of the sampling procedure.

        Args:
            Backbone sampling:
                samples (int, optional): The number of proteins to sample.
                    Default is 1.
                steps (int, optional): The number of integration steps for the SDE.
                    Default is 500.
                chain_lengths (List[int], optional): The lengths of the protein chains.
                    Default is [100].
                conditioner (Conditioner, optional): The conditioner object that
                    provides the conditioning information. Default is None.
                langevin_isothermal (bool, optional): Whether to use the isothermal
                    version of the Langevin SDE. Default is False.
                integrate_func (str, optional): The name of the integration function to
                    use. Default is “euler_maruyama”.
                sde_func (str, optional): The name of the SDE function to use. Defaults
                    to “reverse_sde”.
                langevin_factor (float, optional): The factor that controls the strength
                    of the Langevin noise. Default is 2.
                inverse_temperature (float, optional): The inverse temperature parameter
                    for the SDE. Default is 10.
                protein_init (Protein, optional): The initial protein state. Defaults
                    to None.
                full_output (bool, optional): Whether to return the full outputs of the
                    SDE integration, including the protein sample trajectory, the
                    Xhat trajectory (the trajectory of the preceived denoising target)
                    and the Xunc trajectory (the trajectory of the unconditional sample
                    path). Default is False.
                initialize_noise (bool, optional): Whether to initialize the noise for
                    the SDE integration. Default is True.
                tspan (List[float], optional): The time span for the SDE integration.
                    Default is (1.0, 0.001).
                trajectory_length (int, optional): The number of sampled steps in the
                    trajectory output.  Maximum is `steps`. Default 200.
                **kwargs: Additional keyword arguments for the integration function.

            Sequence and sidechain sampling:
                design_ban_S (list of str, optional): List of amino acid single-letter
                    codes to ban, e.g. `["C"]` to ban cysteines.
                design_method (str, optional): Specifies which method to use for design.
                    Can be `potts` and `autoregressive`. Default is `potts`.
                design_selection (str, optional): Clamp selection for
                    conditioning on a subsequence during sequence sampling. Can be
                    1) a PyMOl-like selection string
                        (https://pymolwiki.org/index.php/Property_Selectors)
                    or
                    2) a binary design mask indicating positions with shape `(num_batch,
                    num_residues)`. 1. indicating the residue to be designed and
                    0. indicates the residue will be masked.
                    e.g.
                        design_selection = torch.Tensor([[0., 1. ,1., 0., 1. ... ]])
                    or
                    3) position-specific valid amino acid choices with shape
                    `(num_batch, num_residues, num_alphabet)`.
                design_t (float or torch.Tensor, optional): Diffusion time for models
                    trained with diffusion augmentation of input structures. Setting `t=0`
                    or `t=None` will condition the model to treat the structure as
                    exact coordinates, while values of `t > 0` will condition
                    the model to treat structures as though they were drawn from
                    noise-augmented ensembles with that noise level. For robust design
                    (default) we recommend `t=0.5`, or for literal design we recommend
                    `t=0.0`. May be a float or a tensor of shape `(num_batch)`.
                temperature_S (float, optional): Temperature for sequence sampling.
                    Default 0.01.
                temperature_chi (float, optional): Temperature for chi angle sampling.
                    Default 1e-3.
                top_p_S (float, optional): Top-p sampling cutoff for autoregressive
                    sampling.
                regularization (str, optional): Complexity regularization for
                    sampling.
                potts_mcmc_depth (int, optional): Depth of sampling (number of steps per
                    alphabet letter times number of sites) per cycle.
                potts_proposal (str): MCMC proposal for Potts sampling. Currently implemented
                    proposals are `dlmc` (default) for Discrete Langevin Monte Carlo [1] or
                    `chromatic` for graph-colored block Gibbs sampling.
                    [1] Sun et al. Discrete Langevin Sampler via Wasserstein Gradient Flow (2023).
                potts_symmetry_order (int, optional): Symmetric design.
                    The first `(num_nodes // symmetry_order)` residues in the protein
                    system will be variable, and all consecutively tiled sets of residues
                    will be locked to these during decoding. Internally this is accomplished by
                    summing the parameters Potts model under a symmetry constraint
                    into this reduced sized system and then back imputing at the end.
                    Currently only implemented for Potts models.

        Returns:
            proteins: Sampled `Protein` object or list of  sampled `Protein` objects in
                the case of multiple outputs.
            full_output_dictionary (dict, optional): Additional outputs if
                `full_output=True`.
        """

        # Get KWARGS
        input_args = locals()

        # Dynamically get acceptable kwargs for each method
        backbone_keys = set(inspect.signature(self._sample).parameters)
        design_keys = set(inspect.signature(self.design).parameters)

        # Filter kwargs for each method using dictionary comprehension
        backbone_kwargs = {k: input_args[k] for k in input_args if k in backbone_keys}
        design_kwargs = {k: input_args[k] for k in input_args if k in design_keys}

        # Perform Sampling
        sample_output = self._sample(**backbone_kwargs)

        if full_output:
            protein_sample, output_dictionary = sample_output
        else:
            protein_sample = sample_output
            output_dictionary = None

        # Perform Design
        if design_method is None:
            proteins = protein_sample
        else:
            if isinstance(protein_sample, list):
                proteins = [
                    self.design(protein, **design_kwargs) for protein in protein_sample
                ]
            else:
                proteins = self.design(protein_sample, **design_kwargs)

        # Perform conditioner postprocessing
        if (conditioner is not None) and hasattr(conditioner, "_postprocessing_"):
            proteins, output_dictionary = self._postprocess(
                conditioner, proteins, output_dictionary
            )

        if full_output:
            return proteins, output_dictionary
        else:
            return proteins

    def _postprocess(self, conditioner, proteins, output_dictionary):
        if output_dictionary is None:
            if isinstance(proteins, list):
                proteins = [
                    conditioner._postprocessing_(protein) for protein in proteins
                ]
            else:
                proteins = conditioner._postprocessing_(proteins)
        else:
            if isinstance(proteins, list):
                p_dicts = []
                proteins = []
                for i, protein in enumerate(proteins):
                    p_dict = {}
                    for key, value in output_dictionary.items():
                        p_dict[key] = value[i]

                    protein, p_dict = conditioner._postprocessing_(protein, p_dict)
                    p_dicts.append(p_dict)

                # Merge Output Dictionaries
                output_dictionary = defaultdict(list)
                for p_dict in p_dicts:
                    for k, v in p_dict.items():
                        output_dictionary[k].append(v)
            else:
                proteins, output_dictionary = conditioner._postprocessing_(
                    proteins, output_dictionary
                )
        return proteins, output_dictionary

    def _sample(
        self,
        samples: int = 1,
        steps: int = 500,
        chain_lengths: List[int] = [100],
        tspan: List[float] = (1.0, 0.001),
        protein_init: Protein = None,
        conditioner: Optional[nn.Module] = None,
        langevin_factor: float = 2,
        langevin_isothermal: bool = False,
        inverse_temperature: float = 10,
        initialize_noise: bool = True,
        integrate_func: Literal["euler_maruyama", "heun"] = "euler_maruyama",
        sde_func: Literal["langevin", "reverse_sde", "ode"] = "reverse_sde",
        trajectory_length: int = 200,
        full_output: bool = False,
        **kwargs,
    ) -> Union[
        Tuple[List[Protein], List[Protein]],
        Tuple[List[Protein], List[Protein], List[Protein], List[Protein]],
    ]:
        """Samples backbones given chain lengths by integrating SDEs.

        Args:
            samples (int, optional): The number of proteins to sample. Default is 1.
            steps (int, optional): The number of integration steps for the SDE.
                Default is 500.
            chain_lengths (List[int], optional): The lengths of the protein chains.
                Default is [100].
            conditioner (Conditioner, optional): The conditioner object that provides
                the conditioning information. Default is None.
            langevin_isothermal (bool, optional): Whether to use the isothermal version
                of the Langevin SDE. Default is False.
            integrate_func (str, optional): The name of the integration function to use.
                Default is `euler_maruyama`.
            sde_func (str, optional): The name of the SDE function to use. Default is
                “reverse_sde”.
            langevin_factor (float, optional): The factor that controls the strength of
                the Langevin noise. Default is 2.
            inverse_temperature (float, optional): The inverse temperature parameter
                for the SDE. Default is 10.
            protein_init (Protein, optional): The initial protein state. Default is
                None.
            full_output (bool, optional): Whether to return the full outputs of the SDE
                integration, including Xhat and Xunc. Default is False.
            initialize_noise (bool, optional): Whether to initialize the noise for the
                SDE integration. Default is True.
            tspan (List[float], optional): The time span for the SDE integration.
                Default is (1.0, 0.001).
            trajectory_length (int, optional): The number of sampled steps in the
                trajectory output.  Maximum is `steps`. Default 200.
            **kwargs: Additional keyword arguments for the integration function.

        Returns:
            proteins: Sampled `Protein` object or list of  sampled `Protein` objects in
                the case of multiple outputs.
            full_output_dictionary (dict, optional): Additional outputs if
                `full_output=True`.
        """

        if protein_init is not None:
            X_unc, C_unc, S_unc = protein_init.to_XCS()
        else:
            X_unc, C_unc, S_unc = self._init_backbones(samples, chain_lengths)

        outs = self.backbone_network.sample_sde(
            C_unc,
            X_init=X_unc,
            conditioner=conditioner,
            tspan=tspan,
            langevin_isothermal=langevin_isothermal,
            integrate_func=integrate_func,
            sde_func=sde_func,
            langevin_factor=langevin_factor,
            inverse_temperature=inverse_temperature,
            N=steps,
            initialize_noise=initialize_noise,
            **kwargs,
        )

        if S_unc.shape != outs["C"].shape:
            S = torch.zeros_like(outs["C"]).long()
        else:
            S = S_unc

        assert S.shape == outs["C"].shape

        proteins = [
            Protein.from_XCS(outs_X[None, ...], outs_C[None, ...], outs_S[None, ...])
            for outs_X, outs_C, outs_S in zip(outs["X_sample"], outs["C"], S)
        ]
        if samples == 1:
            proteins = proteins[0]

        if not full_output:
            return proteins
        else:
            outs["S"] = S
            trajectories = self._format_trajectory(
                outs, "X_trajectory", trajectory_length
            )

            trajectories_Xhat = self._format_trajectory(
                outs, "Xhat_trajectory", trajectory_length
            )

            # use unconstrained C and S for Xunc_trajectory
            outs["S"] = S_unc
            outs["C"] = C_unc
            trajectories_Xunc = self._format_trajectory(
                outs, "Xunc_trajectory", trajectory_length
            )

            if samples == 1:
                full_output_dictionary = {
                    "trajectory": trajectories[0],
                    "Xhat_trajectory": trajectories_Xhat[0],
                    "Xunc_trajectory": trajectories_Xunc[0],
                }
            else:
                full_output_dictionary = {
                    "trajectory": trajectories,
                    "Xhat_trajectory": trajectories_Xhat,
                    "Xunc_trajectory": trajectories_Xunc,
                }

            return proteins, full_output_dictionary

    def cg_sample(
        self,
        cg_map,
        cg_target,
            allowed,
            cg_loss_weight,
            fixed=False,
            noise_range=None,
        # Backbone Args
        samples: int = 1,
        steps: int = 500,
        chain_lengths: List[int] = [100],
        tspan: List[float] = (1.0, 0.001),
        protein_init: Protein = None,
        conditioner: Optional[nn.Module] = None,
        langevin_factor: float = 2,
        langevin_isothermal: bool = False,
        inverse_temperature: float = 10,
        initialize_noise: bool = True,
        integrate_func: Literal["euler_maruyama", "heun"] = "euler_maruyama",
        sde_func: Literal["langevin", "reverse_sde", "ode"] = "reverse_sde",
        trajectory_length: int = 200,
        full_output: bool = False,
        # Sidechain Args
        design_ban_S: Optional[List[str]] = None,
        design_method: Literal["potts", "autoregressive"] = "potts",
        design_selection: Optional[Union[str, torch.Tensor]] = None,
        design_t: Optional[float] = 0.5,
        temperature_S: float = 0.01,
        temperature_chi: float = 1e-3,
        top_p_S: Optional[float] = None,
        regularization: Optional[str] = "LCP",
        potts_mcmc_depth: int = 500,
        potts_proposal: Literal["dlmc", "chromatic"] = "dlmc",
        potts_symmetry_order: int = None,
        verbose: bool = False,
    ) -> Union[
        Union[Protein, List[Protein]], Tuple[Union[Protein, List[Protein]], dict]
    ]:
        """
        Performs Backbone Sampling and Sequence Design and returns a Protein or list
        of Proteins. Optionally this method can return additional arguments to show
        details of the sampling procedure.

        Args:
            Backbone sampling:
                samples (int, optional): The number of proteins to sample.
                    Default is 1.
                steps (int, optional): The number of integration steps for the SDE.
                    Default is 500.
                chain_lengths (List[int], optional): The lengths of the protein chains.
                    Default is [100].
                conditioner (Conditioner, optional): The conditioner object that
                    provides the conditioning information. Default is None.
                langevin_isothermal (bool, optional): Whether to use the isothermal
                    version of the Langevin SDE. Default is False.
                integrate_func (str, optional): The name of the integration function to
                    use. Default is “euler_maruyama”.
                sde_func (str, optional): The name of the SDE function to use. Defaults
                    to “reverse_sde”.
                langevin_factor (float, optional): The factor that controls the strength
                    of the Langevin noise. Default is 2.
                inverse_temperature (float, optional): The inverse temperature parameter
                    for the SDE. Default is 10.
                protein_init (Protein, optional): The initial protein state. Defaults
                    to None.
                full_output (bool, optional): Whether to return the full outputs of the
                    SDE integration, including the protein sample trajectory, the
                    Xhat trajectory (the trajectory of the preceived denoising target)
                    and the Xunc trajectory (the trajectory of the unconditional sample
                    path). Default is False.
                initialize_noise (bool, optional): Whether to initialize the noise for
                    the SDE integration. Default is True.
                tspan (List[float], optional): The time span for the SDE integration.
                    Default is (1.0, 0.001).
                trajectory_length (int, optional): The number of sampled steps in the
                    trajectory output.  Maximum is `steps`. Default 200.
                **kwargs: Additional keyword arguments for the integration function.

            Sequence and sidechain sampling:
                design_ban_S (list of str, optional): List of amino acid single-letter
                    codes to ban, e.g. `["C"]` to ban cysteines.
                design_method (str, optional): Specifies which method to use for design.
                    Can be `potts` and `autoregressive`. Default is `potts`.
                design_selection (str, optional): Clamp selection for
                    conditioning on a subsequence during sequence sampling. Can be
                    1) a PyMOl-like selection string
                        (https://pymolwiki.org/index.php/Property_Selectors)
                    or
                    2) a binary design mask indicating positions with shape `(num_batch,
                    num_residues)`. 1. indicating the residue to be designed and
                    0. indicates the residue will be masked.
                    e.g.
                        design_selection = torch.Tensor([[0., 1. ,1., 0., 1. ... ]])
                    or
                    3) position-specific valid amino acid choices with shape
                    `(num_batch, num_residues, num_alphabet)`.
                design_t (float or torch.Tensor, optional): Diffusion time for models
                    trained with diffusion augmentation of input structures. Setting `t=0`
                    or `t=None` will condition the model to treat the structure as
                    exact coordinates, while values of `t > 0` will condition
                    the model to treat structures as though they were drawn from
                    noise-augmented ensembles with that noise level. For robust design
                    (default) we recommend `t=0.5`, or for literal design we recommend
                    `t=0.0`. May be a float or a tensor of shape `(num_batch)`.
                temperature_S (float, optional): Temperature for sequence sampling.
                    Default 0.01.
                temperature_chi (float, optional): Temperature for chi angle sampling.
                    Default 1e-3.
                top_p_S (float, optional): Top-p sampling cutoff for autoregressive
                    sampling.
                regularization (str, optional): Complexity regularization for
                    sampling.
                potts_mcmc_depth (int, optional): Depth of sampling (number of steps per
                    alphabet letter times number of sites) per cycle.
                potts_proposal (str): MCMC proposal for Potts sampling. Currently implemented
                    proposals are `dlmc` (default) for Discrete Langevin Monte Carlo [1] or
                    `chromatic` for graph-colored block Gibbs sampling.
                    [1] Sun et al. Discrete Langevin Sampler via Wasserstein Gradient Flow (2023).
                potts_symmetry_order (int, optional): Symmetric design.
                    The first `(num_nodes // symmetry_order)` residues in the protein
                    system will be variable, and all consecutively tiled sets of residues
                    will be locked to these during decoding. Internally this is accomplished by
                    summing the parameters Potts model under a symmetry constraint
                    into this reduced sized system and then back imputing at the end.
                    Currently only implemented for Potts models.

        Returns:
            proteins: Sampled `Protein` object or list of  sampled `Protein` objects in
                the case of multiple outputs.
            full_output_dictionary (dict, optional): Additional outputs if
                `full_output=True`.
        """

        # Get KWARGS
        input_args = locals()
        if noise_range is None:
            self.backbone_network.noise_perturb.noise_schedule.log_snr_range = (0,3)
        else:
            self.backbone_network.noise_perturb.noise_schedule.log_snr_range = (noise_range[0], noise_range[1])
        noise = self.backbone_network.noise_perturb.noise_schedule

        from chroma.layers.structure.conditioners import CGCoordinateConditioner, CGCoordinateFixedConditioner
        if not fixed:
            cg_conditioner = CGCoordinateConditioner(cg_map, cg_target, allowed, noise, cg_loss_weight, device=next(self.parameters()).device)
        else:
            cg_conditioner = CGCoordinateFixedConditioner(cg_map, cg_target, allowed, noise, cg_loss_weight, device=next(self.parameters()).device)
        # Dynamically get acceptable kwargs for each method
        backbone_keys = set(inspect.signature(self._sample).parameters)
        design_keys = set(inspect.signature(self.design).parameters)
        #cg_keys = set(inspect.signature(CGCoordinateConditioner).parameters)

        # Filter kwargs for each method using dictionary comprehension
        backbone_kwargs = {k: input_args[k] for k in input_args if k in backbone_keys}
        design_kwargs = {k: input_args[k] for k in input_args if k in design_keys}
        #cg_kwargs = {k: input_args[k] for k in input_args if k in cg_keys}

        # Perform Sampling

        sample_output = self._sample(protein_init=protein_init, steps=steps, conditioner=cg_conditioner,
                                     inverse_temperature=inverse_temperature, initialize_noise=initialize_noise,
                                     sde_func=sde_func)

        #sample_output = self._sample(protein_init=protein_init, steps=steps,
         #                            inverse_temperature=inverse_temperature, initialize_noise=initialize_noise,
         #                            sde_func=sde_func)

        if full_output:
            protein_sample, output_dictionary = sample_output
        else:
            protein_sample = sample_output
            output_dictionary = None

        proteins = protein_sample

        # Perform conditioner postprocessing
        if (conditioner is not None) and hasattr(conditioner, "_postprocessing_"):
            proteins, output_dictionary = self._postprocess(
                conditioner, proteins, output_dictionary
            )

        if full_output:
            return proteins, output_dictionary
        else:
            return proteins

    def _postprocess(self, conditioner, proteins, output_dictionary):
        if output_dictionary is None:
            if isinstance(proteins, list):
                proteins = [
                    conditioner._postprocessing_(protein) for protein in proteins
                ]
            else:
                proteins = conditioner._postprocessing_(proteins)
        else:
            if isinstance(proteins, list):
                p_dicts = []
                proteins = []
                for i, protein in enumerate(proteins):
                    p_dict = {}
                    for key, value in output_dictionary.items():
                        p_dict[key] = value[i]

                    protein, p_dict = conditioner._postprocessing_(protein, p_dict)
                    p_dicts.append(p_dict)

                # Merge Output Dictionaries
                output_dictionary = defaultdict(list)
                for p_dict in p_dicts:
                    for k, v in p_dict.items():
                        output_dictionary[k].append(v)
            else:
                proteins, output_dictionary = conditioner._postprocessing_(
                    proteins, output_dictionary
                )
        return proteins, output_dictionary

    def _format_trajectory(self, outs, key, trajectory_length):
        trajectories = [
            Protein.from_XCS_trajectory(
                [
                    outs_X[i][None, ...]
                    for outs_X in self._resample_trajectory(
                        trajectory_length, outs[key]
                    )
                ],
                outs_C[None, ...],
                outs_S[None, ...],
            )
            for i, (outs_C, outs_S) in enumerate(zip(outs["C"], outs["S"]))
        ]
        return trajectories

    def _resample_trajectory(self, trajectory_length, trajectory):
        if trajectory_length < 0:
            raise ValueError(
                "The trajectory length must fall on the interval [0, sample_steps]."
            )
        n = len(trajectory)
        trajectory_length = min(n, trajectory_length)
        idx = torch.linspace(0, n - 1, trajectory_length).long()
        return [trajectory[i] for i in idx]

    def design(
        self,
        protein: Protein,
        design_ban_S: Optional[List[str]] = None,
        design_method: Literal["potts", "autoregressive"] = "potts",
        design_selection: Optional[Union[str, torch.Tensor]] = None,
        design_t: Optional[float] = 0.5,
        temperature_S: float = 0.01,
        temperature_chi: float = 1e-3,
        top_p_S: Optional[float] = None,
        regularization: Optional[str] = "LCP",
        potts_mcmc_depth: int = 500,
        potts_proposal: Literal["dlmc", "chromatic"] = "dlmc",
        potts_symmetry_order: Optional[int] = None,
        verbose: bool = False,
    ) -> Protein:
        """Performs sequence design and repacking on the specified Protein object
        and returns an updated copy.

        Args:
            protein (Protein): The protein to design.
            design_ban_S (list of str, optional): List of amino acid single-letter
                codes to ban, e.g. `["C"]` to ban cysteines.
            design_method (str, optional): Specifies which method to use for design. valid
                methods are potts and autoregressive. Default is potts.
            design_selection (str or torch.Tensor, optional): Clamp selection for
                conditioning on a subsequence during sequence sampling. Can be
                either a selection string or a binary design mask indicating
                positions to be sampled with shape `(num_batch, num_residues)` or
                position-specific valid amino acid choices with shape
                `(num_batch, num_residues, num_alphabet)`.
            design_t (float or torch.Tensor, optional): Diffusion time for models
                trained with diffusion augmentation of input structures. Setting `t=0`
                or `t=None` will condition the model to treat the structure as
                exact coordinates, while values of `t > 0` will condition
                the model to treat structures as though they were drawn from
                noise-augmented ensembles with that noise level. For robust design
                (default) we recommend `t=0.5`, or for literal design we recommend
                `t=0.0`. May be a float or a tensor of shape `(num_batch)`.
            temperature_S (float, optional): Temperature for sequence sampling.
                Default 0.01.
            temperature_chi (float, optional): Temperature for chi angle sampling.
                Default 1e-3.
            top_p_S (float, optional): Top-p sampling cutoff for autoregressive
                sampling.
            regularization (str, optional): Complexity regularization for
                sampling.
            potts_mcmc_depth (int, optional): Depth of sampling (number of steps per
                alphabet letter times number of sites) per cycle.
            potts_proposal (str): MCMC proposal for Potts sampling. Currently implemented
                proposals are `dlmc` (default) for Discrete Langevin Monte Carlo [1] or
                `chromatic` for graph-colored block Gibbs sampling.
                [1] Sun et al. Discrete Langevin Sampler via Wasserstein Gradient Flow (2023).
            potts_symmetry_order (int, optional): Symmetric design.
                The first `(num_nodes // symmetry_order)` residues in the protein
                system will be variable, and all consecutively tiled sets of residues
                will be locked to these during decoding. Internally this is accomplished by
                summing the parameters Potts model under a symmetry constraint
                into this reduced sized system and then back imputing at the end.
                Currently only implemented for Potts models.

        Returns:
            A new Protein object with updated sequence and, optionally, side-chains.
        """
        protein = copy.deepcopy(protein)
        protein.canonicalize()

        X, C, S = protein.to_XCS()
        if design_method not in set(["potts", "autoregressive"]):
            raise NotImplementedError(
                "Valid design methods are potts and autoregressive, recieved"
                f" {design_method}"
            )

        # Optional sequence clamping
        mask_sample = None
        if design_selection is not None:
            if isinstance(design_selection, str):
                design_selection = protein.get_mask(design_selection)
            mask_sample = design_selection

        X_sample, S_sample, _ = self.design_network.sample(
            X,
            C,
            S,
            t=design_t,
            mask_sample=mask_sample,
            temperature_S=temperature_S,
            temperature_chi=temperature_chi,
            ban_S=design_ban_S,
            sampling_method=design_method,
            regularization=regularization,
            potts_sweeps=potts_mcmc_depth,
            potts_proposal=potts_proposal,
            verbose=verbose,
            symmetry_order=potts_symmetry_order,
        )
        protein.sys.update_with_XCS(X_sample, C=None, S=S_sample)
        return protein

    def get_potts_energy(self,
        protein: Protein,
        design_ban_S: Optional[List[str]] = None,
        design_method: Literal["potts", "autoregressive"] = "potts",
        design_selection: Optional[Union[str, torch.Tensor]] = None,
        design_t: Optional[float] = 0.5,
        temperature_S: float = 0.01,
        temperature_chi: float = 1e-3,
        top_p_S: Optional[float] = None,
        regularization: Optional[str] = "LCP",
        potts_mcmc_depth: int = 500,
        potts_proposal: Literal["dlmc", "chromatic"] = "dlmc",
        potts_symmetry_order: Optional[int] = None,
        verbose: bool = False,
    ):


        protein = copy.deepcopy(protein)
        protein.canonicalize()
        device = next(self.parameters()).device

        traj_len = protein.sys.num_models()
        batch_size =1
        num_batches = int(traj_len / batch_size )
        batches = [list(range(i*batch_size, (i+1) * batch_size)) for i in range(num_batches)]
        X_traj, C, S = protein.to_XCS_trajectory()

        #print("S", S.shape)
        #import chroma.utility.polyseq as polyseq
        #for i in range(S.shape[-1]):
        #    print(i+1, int(S[0][i]), polyseq.index_to_triple(int(S[0][i])))
        #quit()

        X = torch.zeros((batch_size, *X_traj[0].shape[1:]), device=C.device)
        for i in batches[0]:
            X[i] = X_traj[i]
        #print("X", X.shape)
        #print("X_traj", len(X_traj), X_traj[0].shape)

        #print("X", X.device)
        #print("X_traj", len(X_traj))

        #X = X.expand(traj_len, *X.shape[1:])
        C = C.expand(X.size(0), *C.shape[1:])

        #print("S", S.device)
        #print("C", C.device)


        if design_method not in set(["potts", "autoregressive"]):
            raise NotImplementedError(
                "Valid design methods are potts and autoregressive, recieved"
                f" {design_method}"
            )

        gc.collect()
        torch.cuda.empty_cache()
        mutations = self.generate_single_mutations(S)
        #if X.shape[2] == 4:
        #    X = nn.functional.pad(X, [0, 0, 0, 10])

        #print("node_h", node_h.shape)
        #print("X", X.shape)
        #print("C", C.shape)
        node_h, edge_h, edge_idx, mask_i, mask_ij = self.design_network.encode(X, C, t=0)
        h, J = self.design_network.decoder_S_potts.forward(node_h, edge_h, edge_idx, mask_i, mask_ij)
        U_total = self.potts_from_h_J(h, J, S, edge_idx).sum(1).detach()
        #print("U_total", U_total.shape)
        U_mutate_total = self.potts_from_h_J(h, J, mutations, edge_idx).sum(1).detach()
        #print("U_mutate_total", U_mutate_total.shape)

        a = time()
        for i in range(1, num_batches):
            #print("X", X.shape)
            #print("X_traj", len(X_traj), X_traj[0].shape)
            for ind, q in enumerate(batches[i]):
                X[ind] = X_traj[q]
            node_h, edge_h, edge_idx, mask_i, mask_ij = self.design_network.encode(X, C, t=0)
            h, J = self.design_network.decoder_S_potts.forward(node_h, edge_h, edge_idx, mask_i, mask_ij)
            U = self.potts_from_h_J(h, J, S, edge_idx)
            U_mutate = self.potts_from_h_J(h,J,mutations, edge_idx).sum(1).detach()
            U_total += U.sum(1).detach()
            U_mutate_total = torch.add(U_mutate.detach(), U_mutate_total.detach())
            del h, J, edge_idx
            gc.collect()
            torch.cuda.empty_cache()
            #U_mean = U.mean(dim=1)
            #U_mutate_mean = U_mutate.mean(dim=1)
            #print(i,time()-a)
        U_mean = torch.divide(U_total, num_batches * batch_size)
        U_mutate_mean = torch.divide(U_mutate_total, num_batches * batch_size)

        return S[0].cpu().numpy(), mutations.cpu().numpy(), torch.subtract(U_mutate_mean, U_mean).cpu().numpy()

    def get_potts_energy_derivatives_by_bin(self,
                         protein: Protein,
                         design_method: Literal["potts", "autoregressive"] = "potts",
                         cg_frame_weight: Optional[torch.Tensor] = None,
                         cg_bin_indices: Optional[torch.Tensor] = None,
                         mutations: Optional[torch.Tensor]=None,
                         batch_size: Optional[int] = 1
                         ):

        protein = copy.deepcopy(protein)
        protein.canonicalize()
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = next(self.parameters()).device

        traj_len = protein.sys.num_models()

        if cg_bin_indices is None:
            cg_bin_indices = torch.randint(low=0, high=5, size=(traj_len,))
        else:
            cg_bin_indices.to(device)

        cg_bin_indices = cg_bin_indices.to(device)

        n_bins = int(cg_bin_indices.max() + 1)

        num_batches = int(traj_len / batch_size)
        batches = [list(range(i * batch_size, (i + 1) * batch_size)) for i in range(num_batches)]
        X_traj, C, S = protein.to_XCS_trajectory()

        #print("S", S.shape)
        #print("cg_bin_indices", cg_bin_indices.shape, cg_bin_indices.device)
        #print("cg_frame_weight", cg_frame_weight)


        X = torch.zeros((batch_size, *X_traj[0].shape[1:]), device=C.device)
        for i in batches[0]:
            X[i] = X_traj[i]
        C = C.expand(X.size(0), *C.shape[1:])

        if design_method not in set(["potts", "autoregressive"]):
            raise NotImplementedError(
                "Valid design methods are potts and autoregressive, recieved"
                f" {design_method}"
            )

        if mutations is None:
            mutations = self.generate_single_mutations(S)
        else:
            mutations = mutations.to(device)
        #gc.collect()
        #torch.cuda.empty_cache()
        #print("mutations", mutations.shape, mutations.device, mutations)
        #quit()
        if cg_frame_weight is None:
            cg_frame_weight = torch.ones(traj_len)
        #cg_frame_weight = torch.Tensor([1])
        cg_frame_weight = cg_frame_weight.to(device)
        cg_frame_weight = torch.divide(cg_frame_weight, cg_frame_weight.sum())

        #print("cg_frame_weight", cg_frame_weight.device)
        #print("cg_frame_weight", cg_frame_weight)

        # print("X", X.shape)
        # print("C", C.shape)

        node_h, edge_h, edge_idx, mask_i, mask_ij = self.design_network.encode(X, C, t=0)
        h, J = self.design_network.decoder_S_potts.forward(node_h, edge_h, edge_idx, mask_i, mask_ij)
        U_total = self.potts_from_h_J(h, J, S, edge_idx).detach()
        #print("U_total", U_total.shape)
        U_total_weighted = U_total @ cg_frame_weight[batches[0]]
        #print("U_total_weighted", U_total_weighted.shape)

        U_bins = torch.zeros(n_bins, *U_total_weighted.shape[1:], device=device)
        #print("U_bins", U_bins.shape)
        for ind2 in range(batch_size):
            U_bins[cg_bin_indices[ind2]] = torch.add(U_bins[cg_bin_indices[ind2]], U_total[0][ind2])
        #print("U_bins", U_bins.shape)


        U_mutate_total = self.potts_from_h_J(h, J, mutations, edge_idx).detach()
        #print("U_mutate_total", U_mutate_total.shape)
        #print("batches[0]", batches[0])
        #print("cg_frame_weight", cg_frame_weight.shape )
        #print("cg_frame_weight[[0]]", cg_frame_weight[[0]])
        U_mutate_total_weighted = U_mutate_total @ cg_frame_weight[batches[0]]
        #print("U_mutate_total_weighted", U_total_weighted.shape)

        U_mutate_bins = torch.zeros(n_bins, *U_mutate_total_weighted.shape, device=device)
        for ind2 in range(batch_size):
            #print("U_mutate_bins[cg_bin_indices[ind2]]", U_mutate_bins[cg_bin_indices[ind2]].shape)
            #print("U_mutate_total[:,ind2]", U_mutate_total.shape)
            U_mutate_bins[cg_bin_indices[ind2]] = torch.add(U_mutate_bins[cg_bin_indices[ind2]], U_mutate_total[:,ind2])
        #print("U_mutate_bins[:,0]", U_mutate_bins[:,0])

        a = time()
        for i in range(1, num_batches):
            #print("batch number", i)
            # print("X", X.shape)
            # print("X_traj", len(X_traj), X_traj[0].shape)
            for ind, q in enumerate(batches[i]):
                #print("ind, q", ind, q)
                #print("batches[i]", batches[i])
                X[ind] = X_traj[q]
            node_h, edge_h, edge_idx, mask_i, mask_ij = self.design_network.encode(X, C, t=0)
            h, J = self.design_network.decoder_S_potts.forward(node_h, edge_h, edge_idx, mask_i, mask_ij)
            U_total = self.potts_from_h_J(h, J, S, edge_idx).detach()
            U_total_weighted = torch.add(U_total_weighted, U_total @ cg_frame_weight[batches[i]])
            for ind, ind2 in enumerate(batches[i]):
                U_bins[cg_bin_indices[ind2]] = torch.add(U_bins[cg_bin_indices[ind2]], U_total[0][ind])
                #print("U_bins i", U_bins)


            U_mutate_total = self.potts_from_h_J(h, J, mutations, edge_idx).detach()
            U_mutate_total_weighted = torch.add(U_mutate_total_weighted, U_mutate_total @ cg_frame_weight[batches[i]])
            for ind, ind2 in enumerate(batches[i]):
                #print("ind, ind2", ind, ind2)
                #print("cg_bin_indices[ind2]", cg_bin_indices[ind2])
                U_mutate_bins[cg_bin_indices[ind2]] = torch.add(U_mutate_bins[cg_bin_indices[ind2]], U_mutate_total[:, ind])
                #print("U_mutate_bins[:,0] i", U_mutate_bins[:,0])

            del h, J, edge_idx
            gc.collect()
            torch.cuda.empty_cache()
            #print(i, time() - a)

        #print("U_mutate_bins", U_mutate_bins.shape)
        #print("U_bins", U_bins.shape)
        term1 = torch.subtract(U_mutate_bins, U_bins.unsqueeze(1))
        #print("term 1", term1.shape)

        values, counts = torch.unique(cg_bin_indices, return_counts=True)

        print("values", values)
        print("counts", counts)
        print("cg_bin_indices", cg_bin_indices.shape)
        print("term1", term1.shape)
        term1_mean = term1 / counts.unsqueeze(1)



        print("term1_mean", term1_mean.shape)
        print("term1_mean.T[0]", term1_mean.T[0])

        return S[0].cpu().numpy(), mutations.cpu().numpy(), term1_mean.cpu().numpy()

    def potts_from_h_J(self, h,J,S, edge_idx):


        #print("s to start", S.shape)
        #print("h", h.shape)

        struct_dim = copy.deepcopy(h.size(0))

        #h = h.unsqueeze(0).expand(S.size(0), *h.shape).reshape(S.size(0)*h.size(0), *h.shape[1:])
        h = h.unsqueeze(0).expand(S.size(0), *h.shape)
        #print("new h", h.shape)

        #J = J.unsqueeze(0).expand(S.size(0), *J.shape).reshape(S.size(0)*J.size(0), *J.shape[1:])
        J = J.unsqueeze(0).expand(S.size(0), *J.shape)
        #J = J.expand(S.size(0), *J.shape[1:])
        #edge_idx = edge_idx.expand(S.size(0), *edge_idx.shape[1:])
        edge_idx = edge_idx.unsqueeze(0).expand(S.size(0), *edge_idx.shape).reshape(S.size(0)*edge_idx.size(0), *edge_idx.shape[1:])
        #edge_idx = edge_idx.unsqueeze(0).expand(S.size(0), *edge_idx.shape)
        #print("new j", J.shape)
        #print("new edge_idx", edge_idx.shape)
        #print("s", S.shape)
        old_shape = copy.deepcopy(S.shape)
        S = (S.unsqueeze(1).expand(S.size(0),struct_dim,*S.shape[1:]))
        S = S.reshape(edge_idx.size(0), *S.shape[2:])
        #print("new s", S.shape)
        #S = S.unsqueeze(1).expand(S.size(0), struct_dim, *S.shape[1:])




        S_j = graph.collect_neighbors(S.unsqueeze(-1), edge_idx)
        #print("S_j", S_j.shape)

        S_j = S_j.unsqueeze(-1).expand(-1, -1, -1, h.shape[-1], -1)
        #print(S_j.shape)

        S_j = S_j.unsqueeze(0).reshape(J.size(0), int(S.size(0)/ J.size(0)),*S_j.shape[1:])

        #print("S_j", S_j.shape)
        #print("J", J.shape)
        #quit()
        J_ij = torch.gather(J, -1, S_j).squeeze(-1)
        #print("J_ij", J_ij.shape)
        # Sum out J contributions to yield local conditionals
        J_i = J_ij.sum(3)
        #print("J_i", J_i.shape)
        #print("h", h.shape)
        U_i = h + J_i
        #print(J_i.shape)
        #print(U_i.shape)

        #print("final")
        #print("U_i", U_i.shape)
        #print("J_i", J_i.shape)


        # Correct for double counting in total energy
        #print("S", S.shape)
        S_expand = S[..., None]
        #S_expand = S_expand.unsqueeze(1).expand(S_expand.size(0), struct_dim, *S_expand.shape[1:])

        S_expand = S_expand.unsqueeze(0).reshape(J_i.size(0), int(S_expand.size(0) / J_i.size(0)), *S_expand.shape[1:])
        #print("S_expand", S_expand.shape)
        U = (
                torch.gather(U_i, -1, S_expand) - 0.5 * torch.gather(J_i, -1, S_expand)
        ).sum((2, 3))

        return U

    def generate_single_mutations(self, S):

        device = S.device

        L = S.size(1)
        mutations = S.repeat(L*20, 1)

        #new_values = torch.arange(20, device=device).repeat(L)
        #target_indices = torch.arange(L, device=device).repeat_interleave(20)
        #mutations[torch.arange(L * 20), target_indices] = new_values

        #n = 2
        #start =546
        start =0
        #n= 217
        n=L
        new_values = torch.arange(20, device=device).repeat(n)
        target_indices = torch.arange(start, start+n,1, device=device).repeat_interleave(20)

        #print("mutations", mutations.shape)

        #print("target ind")
        mutations[torch.arange(n*20), target_indices] = new_values

        mask = ~(mutations==S).all(dim=1)
        return mutations[mask]

    def generate_trimer_mutations(self, S):

        device = S.device

        L = S.size(1)
        mutations = S.repeat(L * 20, 1)



        #start = 546
        start =0
        n = 182
        # n=L
        new_values = torch.arange(20, device=device).repeat(n)

        target_indices = torch.arange(start + n*0, start + n*1, 1, device=device).repeat_interleave(20)
        mutations[torch.arange(n * 20), target_indices] = new_values

        target_indices = torch.arange(start + n*1, start + n*2, 1, device=device).repeat_interleave(20)
        mutations[torch.arange(n * 20), target_indices] = new_values

        target_indices = torch.arange(start + n * 2, start + n * 3, 1, device=device).repeat_interleave(20)
        mutations[torch.arange(n * 20), target_indices] = new_values


        mask = ~(mutations == S).all(dim=1)
        return mutations[mask]


    def _design_ar(self, protein, alphabet=None, temp_S=0.1, temp_chi=1e-3):
        X, C, S = protein.to_XCS()
        ban_S = None
        if alphabet is not None:
            ban_S = set(AA20_3).difference(alphabet)

        X_sample, S_sample, _, _ = self.design_network_ar.sample(
            X,
            C,
            S,
            temperature_S=temp_S,
            temperature_chi=temp_chi,
            return_scores=True,
            ban_S=ban_S,
        )

        protein.sys.update_with_XCS(X_sample, C=None, S=S_sample)

        return protein

    def pack(
        self, protein: Protein, temperature_chi: float = 1e-3, clamped: bool = False
    ) -> Protein:
        """Packs Sidechains of a Protein using the design network

        Args:
            protein (Protein): The Protein to repack.
            temperature_chi (float): Temperature parameter for sampling chi
                angles. Even if a high temperature sequence is sampled, this is
                recommended to always be low. Default is `1E-3`.
            clamped (bool): If `True`, no sampling is done and the likelihood
                values will be calculated for the input sequence and structure.
                Used for validating the sequential versus parallel decoding
                modes. Default is `False`

        Returns:
            Protein: The Repacked Protein
        """
        X, C, S = protein.to_XCS(all_atom=False)

        X_repack, _, _ = self.design_network.pack(
            X,
            C,
            S,
            temperature_chi=temperature_chi,
            clamped=clamped,
            return_scores=True,
        )
        # Convert S_repack to seq
        protein.sys.update_with_XCS(X_repack, C=None, S=S)

        return protein

    def score_backbone(
        self,
        proteins: Union[List[Protein], Protein],
        num_samples: int = 50,
        tspan: List[float] = [1e-4, 1.0],
    ) -> Union[List[dict], dict]:
        """
        Score Proteins with the following chroma scores:
            elbo:
            elbo_X:
            rmsd_ratio:
            fragment_mse:
            neighborhood_mse:
            distance_mse:
            hb_local:
            hb_nonlocal:

        Args:
            proteins (list of Protein or Protein): The Proteins to be scored.
            num_samples (int, optional): The number of time points to calculate the metrics. Default 50.
            tspan (list of float, optional): A list of two times [t_initial, t_final] which represent
                the range of times to draw samples. Default [1e-4, 1.0].

        Returns:
            List of dict or dict: A dictionary containing all of the score data.
            Scores are returned as named tuples.
        """

        # Extract XCS for scoring
        device = next(self.parameters()).device
        if isinstance(proteins, list):
            X, C, S = self._protein_list_to_XCS(proteins, device=device)
        else:
            X, C, S = proteins.to_XCS(device=device)

        # Generate Scores
        metrics, metrics_samples = self.backbone_network.estimate_metrics(
            X, C, return_samples=True, num_samples=num_samples, tspan=tspan
        )

        if isinstance(proteins, list):
            metric_dictionary = [
                self._make_metric_dictionary(metrics, metrics_samples, idx=i)
                for i in range(len(proteins))
            ]
        else:
            metric_dictionary = self._make_metric_dictionary(metrics, metrics_samples)

        return metric_dictionary

    def score_sequence(
        self,
        proteins: Union[List[Protein], Protein],
        t: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Scores designed Proteins with the following Chroma scores:
            - -log(p) for sequences and chi angles
            - average RMSD and number of clashes per side-chain
        For further details on the scores computed, see
            chroma.models.graph_design.GraphDesign.loss.

        Args:
            proteins (list of Protein or Protein): The Proteins to be scored.
            t (torch.Tensor, optional): Diffusion timesteps corresponding to
                noisy input backbones, of shape `(num_batch)`. Default is no
                noise.

        Returns:
            List of dict or dict: A dictionary containing all of the score data.
            Scores are returned as named tuples.
        """

        # Extract XCS for scoring
        device = next(self.parameters()).device
        if isinstance(proteins, list):
            X, C, S = self._protein_list_to_XCS(proteins, all_atom=True, device=device)
            output_scores = [{} for _ in range(len(proteins))]
        else:
            X, C, S = proteins.to_XCS(all_atom=True, device=device)
            output_scores = {}
        losses = self.design_network.loss(X, C, S, t=t, batched=False)
        # each value in the losses dictionary contains the results for all proteins
        for name, loss_tensor in losses.items():
            loss_list = [_t.squeeze() for _t in loss_tensor.split(1)]
            if isinstance(proteins, list):
                for i, loss in enumerate(loss_list):
                    output_scores[i][name] = loss
            else:
                output_scores[name] = loss_list[0]
        return output_scores

    def _protein_list_to_XCS(self, list_of_proteins, all_atom=False, device=None):
        """Package up proteins with padding"""

        # get all the XCS stuff
        Xs, Cs, Ss = zip(
            *[protein.to_XCS(all_atom=all_atom) for protein in list_of_proteins]
        )

        # Get Max Dims for Xs, Cs, Ss
        Dmax = max([C.shape[1] for C in Cs])
        device = Xs[0].device

        # Augment each with zeros
        with torch.no_grad():
            X = torch.cat(
                [nn.functional.pad(X, (0, 0, 0, 0, 0, Dmax - X.shape[1])) for X in Xs]
            )
            C = torch.cat([nn.functional.pad(C, (0, Dmax - C.shape[1])) for C in Cs])
            S = torch.cat([nn.functional.pad(S, (0, Dmax - S.shape[1])) for S in Ss])
        return X, C, S

    def score(
        self,
        proteins: Union[List[Protein], Protein],
        num_samples: int = 50,
        tspan: List[float] = [1e-4, 1.0],
    ) -> Tuple[Union[List[dict], dict], dict]:
        backbone_scores = self.score_backbone(proteins, num_samples, tspan)
        sequence_scores = self.score_sequence(proteins)
        if isinstance(proteins, list):
            for ss in sequence_scores:
                ss["t_seq"] = ss.pop("t")
            return [bs | ss for bs, ss in zip(backbone_scores, sequence_scores)]
        else:
            sequence_scores["t_seq"] = sequence_scores.pop("t")
            return backbone_scores | sequence_scores

    def _make_metric_dictionary(self, metrics, metrics_samples, idx=None):
        # Process Metrics into a Single Dictionary
        metric_dictionary = {}
        for k, vs in metrics_samples.items():
            if k == "t":
                metric_dictionary["t"] = vs
            elif k in ["X", "X0_pred"]:
                if idx is None:
                    v = metrics[k]
                else:
                    vs = vs[idx]
                    v = metrics[k][idx]
                score = namedtuple(k, ["value", "samples"])
                metric_dictionary[k] = score(value=v, samples=vs)
            else:
                if idx is None:
                    v = metrics[k].item()
                else:
                    vs = vs[idx]
                    v = metrics[k][idx].item()
                vs = [i.item() for i in vs]
                score = namedtuple(k, ["score", "subcomponents"])
                metric_dictionary[k] = score(score=v, subcomponents=vs)

        return metric_dictionary

    def _init_backbones(self, num_backbones, length_backbones):
        # Start with purely alpha backbones
        X = ProteinBackbone(
            num_batch=num_backbones,
            num_residues=sum(length_backbones),
            init_state="alpha",
        )()
        C = torch.cat(
            [torch.full([rep], i + 1) for i, rep in enumerate(length_backbones)]
        ).expand(X.shape[0], -1)
        S = torch.zeros_like(C)
        return [i.to(next(self.parameters()).device) for i in [X, C, S]]
