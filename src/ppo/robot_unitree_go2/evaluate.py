"""
Evaluación de política PPO entrenada para Unitree Go2.
Corre el rollout y guarda un MP4 en la carpeta del proyecto.

Estructura esperada del proyecto:
  go2_render/
  ├── evaluate.py                        (este script)
  ├── go2_ppo_1000000_steps.zip
  ├── go2_vecnormalize_200000_steps.pkl
  └── mujoco_menagerie/
      └── unitree_go2/
          └── scene.xml
"""

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import imageio

# ── Configuración ──────────────────────────────────────────────────────────
MODEL_PATH      = 'mujoco_menagerie/unitree_go2/scene.xml'
#POLICY_PATH     = 'go2_ppo_1000000_steps.zip'
POLICY_PATH     = 'go2_ppo_3000000_steps.zip'
#VECNORM_PATH    = 'go2_vecnormalize_200000_steps.pkl'
VECNORM_PATH    = 'go2_vecnormalize_3m.pkl'
OUTPUT_VIDEO    = 'go2_eval_3m.mp4'
EVAL_STEPS      = 500
RENDER_WIDTH    = 640
RENDER_HEIGHT   = 480
FPS             = 50
# ───────────────────────────────────────────────────────────────────────────


class Go2EvalEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': FPS}

    def __init__(self):
        super().__init__()
        self.model      = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data       = mujoco.MjData(self.model)
        self.ctrl_range = self.model.actuator_ctrlrange
        self.init_qpos  = self.model.keyframe('home').qpos.copy() if self.model.nkey > 0 else None

        obs_dim = self._get_obs().shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            self.ctrl_range[:, 0].astype(np.float32),
            self.ctrl_range[:, 1].astype(np.float32),
            dtype=np.float32
        )
        self._step_count = 0

        # Renderer con OpenGL local (funciona en Xubuntu sin configuración extra)
        self.renderer = mujoco.Renderer(self.model, height=RENDER_HEIGHT, width=RENDER_WIDTH)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos[7:],
            self.data.qvel,
            self.data.qpos[3:7]
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        if self.init_qpos is not None:
            self.data.qpos[:] = self.init_qpos
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        terminated = bool(self.data.qpos[2] < 0.18)
        truncated  = self._step_count >= EVAL_STEPS
        return self._get_obs(), 0.0, terminated, truncated, {}

    #def render(self):
    #    self.renderer.update_scene(self.data, camera=0)
    #    return self.renderer.render()

    def render(self):
        self.renderer.update_scene(self.data)  # sin camera=0
        return self.renderer.render()

    def close(self):
        self.renderer.close()


def main():
    print('Cargando entorno...')
    eval_env = DummyVecEnv([Go2EvalEnv])
    eval_env = VecNormalize.load(VECNORM_PATH, eval_env)
    eval_env.training    = False
    eval_env.norm_reward = False

    print('Cargando política...')
    model = PPO.load(POLICY_PATH, env=eval_env, device='cpu')

    print(f'Generando rollout ({EVAL_STEPS} pasos máx)...')
    frames = []
    obs    = eval_env.reset()

    for step in range(EVAL_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = eval_env.step(action)
        frame = eval_env.envs[0].render()
        frames.append(frame)
        #if done[0]:
        #    print(f'  Robot caído en paso {step + 1}')
        #    break

    eval_env.close()

    print(f'Guardando video: {OUTPUT_VIDEO}')
    imageio.mimsave(OUTPUT_VIDEO, frames, fps=FPS, quality=8)
    print(f'✅ Listo — {len(frames)} frames, {len(frames)/FPS:.1f}s')
    print(f'   Video guardado en: {os.path.abspath(OUTPUT_VIDEO)}')


if __name__ == '__main__':
    main()
