import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from policy.irpo import IRPO_Learner
from policy.irpo_blend import IRPO_Blend_Learner
from policy.irpo_is import IRPO_IS_Learner
from policy.irpo_lr import IRPO_LR_Learner
from policy.irpo_pick import IRPO_Pick_Learner
from policy.layers.ppo_networks import PPO_Actor, PPO_Critic
from trainer.irpo_trainer import IRPOTrainer
from utils.intrinsic_rewards import IntrinsicRewardFunctions
from utils.sampler import OnlineSampler


class IRPO_Algorithm(nn.Module):
    def __init__(self, env, logger, writer, args):
        super(IRPO_Algorithm, self).__init__()

        # === Parameter saving === #
        self.env = env
        self.logger = logger
        self.writer = writer
        self.args = args

        self.intrinsic_reward_fn = IntrinsicRewardFunctions(
            logger=logger,
            writer=writer,
            args=args,
        )

        self.current_timesteps = self.intrinsic_reward_fn.current_timesteps

    def begin_training(self):
        # === Sampler === #
        sampler = OnlineSampler(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            episode_len=self.env.max_steps,
            batch_size=self.args.batch_size,
        )

        # === Meta-train using options === #'
        self.define_outer_policy()
        trainer = IRPOTrainer(
            env=self.env,
            policy=self.policy,
            sampler=sampler,
            logger=self.logger,
            writer=self.writer,
            init_timesteps=self.current_timesteps,
            timesteps=self.args.timesteps,
            log_interval=self.args.log_interval,
            eval_num=self.args.eval_num,
            rendering=self.args.rendering,
            seed=self.args.seed,
            args=self.args,
        )
        final_steps = trainer.train()
        self.current_timesteps += final_steps

    def define_outer_policy(self):
        # === Define policy === #
        actor = PPO_Actor(
            input_dim=self.args.state_dim,
            hidden_dim=self.args.actor_fc_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
            device=self.args.device,
        )
        critic = PPO_Critic(self.args.state_dim, hidden_dim=self.args.critic_fc_dim)

        # irpo, irpo_pick, irpo_is, irpo_blend
        if self.args.irpo_type == "irpo":
            # NOTE: THIS FOLLOWS THE PARAMETERS DEFINED IN ARGS
            IRPO = IRPO_Learner

        elif self.args.irpo_type == "irpo_pick":
            # NOTE: This is IRPO with pick (argmax) gradient aggregation
            # NOTE: Specifically, evaluation is performed with the exploratory policy with highest weight.
            IRPO = IRPO_Pick_Learner
            self.args.weight_option = "argmax"  # force argmax
            print("[WARNING] Using IRPO Pick. This forces argmax gradient aggregation.")

        elif self.args.irpo_type == "irpo_blend":
            # NOTE: ONLY args.irm_method = {"zero", "linear", "exponential", "drift" } matters
            # NOTE: Target needs to be defined, which denotes where it ends up using the true gradient.
            IRPO = IRPO_Blend_Learner
            self.args.weight_option = "softmax"  # force softmax
            print(
                "[WARNING] Using IRPO Blend. This forces softmax gradient aggregation."
            )
            assert self.args.irm_method in ["zero", "linear", "exponential", "drift"]

        elif self.args.irpo_type == "irpo_lr":
            IRPO = IRPO_LR_Learner
            self.args.weight_option = "softmax"  # force softmax
            print("[WARNING] Using IRPO LR. This forces softmax gradient aggregation.")

        elif self.args.irpo_type == "irpo_is":
            IRPO = IRPO_IS_Learner
            self.args.weight_option = "softmax"  # force softmax
            print("[WARNING] Using IRPO IS. This forces softmax gradient aggregation.")

        self.policy = IRPO(
            actor=actor,
            critic=critic,
            intrinsic_reward_fn=self.intrinsic_reward_fn,
            timesteps=self.args.timesteps,
            irm_target=self.args.timesteps * 0.1,  # only for irpo_is
            irm_method=self.args.irm_method,
            num_inner_updates=self.args.num_inner_updates,
            outer_level_update_mode=self.args.outer_level_update_mode,
            outer_actor_lr=self.args.outer_actor_lr,
            inner_actor_lr=self.args.inner_actor_lr,
            critic_lr=self.args.critic_lr,
            weight_option=self.args.weight_option,
            entropy_scaler=self.args.entropy_scaler,
            target_kl=self.args.target_kl,
            gamma=self.args.gamma,
            gae=self.args.gae,
            device=self.args.device,
        )

        if hasattr(self.env, "get_grid"):
            self.policy.actor.grid = self.env.get_grid()
