""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import collections
import gym
import numpy as np

from mj_envs.envs.multi_task.multi_task_base_v1 import KitchenBase
from mj_envs.utils.quat_math import euler2quat
from mujoco_py.modder import TextureModder, LightModder
import PIL.Image
import os
from mujoco_py import cymj
from glob import glob 
import random
import mujoco_py
from copy import deepcopy

# ToDo: Get these details from key_frame
DEMO_RESET_QPOS = np.array(
    [
        1.01020992e-01,
        -1.76349747e00,
        1.88974607e00,
        -2.47661710e00,
        3.25189114e-01,
        8.29094410e-01,
        1.62463629e00,
        3.99760380e-02,
        3.99791002e-02,
        2.45778156e-05,
        2.95590127e-07,
        2.45777410e-05,
        2.95589217e-07,
        2.45777410e-05,
        2.95589217e-07,
        2.45777410e-05,
        2.95589217e-07,
        2.16196258e-05,
        5.08073663e-06,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        -2.68999994e-01,
        3.49999994e-01,
        1.61928391e00,
        6.89039584e-19,
        -2.26122120e-05,
        -8.87580375e-19,
    ]
)
DEMO_RESET_QVEL = np.array(
    [
        -1.24094905e-02,
        3.07730486e-04,
        2.10558046e-02,
        -2.11170651e-02,
        1.28676305e-02,
        2.64535546e-02,
        -7.49515183e-03,
        -1.34369839e-04,
        2.50969693e-04,
        1.06229627e-13,
        7.14243539e-16,
        1.06224762e-13,
        7.19794728e-16,
        1.06224762e-13,
        7.21644648e-16,
        1.06224762e-13,
        7.14243539e-16,
        -1.19464428e-16,
        -1.47079926e-17,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        2.93530267e-09,
        -1.99505748e-18,
        3.42031125e-14,
        -4.39396125e-17,
        6.64174740e-06,
        3.52969879e-18,
    ]
)

class KitchenFrankaFixed(KitchenBase):

    OBJ_INTERACTION_SITES = (
        "knob1_site",
        "knob2_site",
        "knob3_site",
        "knob4_site",
        "light_site",
        "slide_site",
        "leftdoor_site",
        "rightdoor_site",
        "microhandle_site",
        "kettle_site0",
        "kettle_site0",
        "kettle_site0",
        "kettle_site0",
        "kettle_site0",
        "kettle_site0",
    )

    OBJ_JNT_NAMES = (
        "knob1_joint",
        "knob2_joint",
        "knob3_joint",
        "knob4_joint",
        "lightswitch_joint",
        "slidedoor_joint",
        "leftdoorhinge",
        "rightdoorhinge",
        "micro0joint",
        "kettle0:Tx",
        "kettle0:Ty",
        "kettle0:Tz",
        "kettle0:Rx",
        "kettle0:Ry",
        "kettle0:Rz",
    )

    ROBOT_JNT_NAMES = (
        "panda0_joint1",
        "panda0_joint2",
        "panda0_joint3",
        "panda0_joint4",
        "panda0_joint5",
        "panda0_joint6",
        "panda0_joint7",
        "panda0_finger_joint1",
        "panda0_finger_joint2",
    )

    def _setup(
        self,
        robot_jnt_names=ROBOT_JNT_NAMES,
        obj_jnt_names=OBJ_JNT_NAMES,
        obj_interaction_site=OBJ_INTERACTION_SITES,
        **kwargs,
    ):
        super()._setup(
            robot_jnt_names=robot_jnt_names,
            obj_jnt_names=obj_jnt_names,
            obj_interaction_site=obj_interaction_site,
            **kwargs,
        )


class KitchenFrankaDemo(KitchenFrankaFixed):

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)

        super()._setup(**kwargs)

    def reset(self, reset_qpos=None, reset_qvel=None):
        if reset_qpos is None:
            reset_qpos = self.init_qpos.copy()
            reset_qvel = self.init_qvel.copy()
            reset_qpos[self.robot_dofs] = DEMO_RESET_QPOS[self.robot_dofs]
            reset_qvel[self.robot_dofs] = DEMO_RESET_QVEL[self.robot_dofs]
        return super().reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel)


class KitchenFrankaRandom(KitchenFrankaFixed):

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)

        super()._setup(**kwargs)

    def reset(self, reset_qpos=None, reset_qvel=None):
        if reset_qpos is None:
            reset_qpos = self.init_qpos.copy()
            reset_qpos[self.robot_dofs] += (
                0.05
                * (self.np_random.uniform(size=len(self.robot_dofs)) - 0.5)
                * (self.robot_ranges[:, 1] - self.robot_ranges[:, 0])
            )
        return super().reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel)

TEXTURE_ID_TO_INFOS = {
    1: dict(
        name='floor', 
        shape=(1024, 1024, 3), 
        group='floor',
    ),
    5: dict(
        name='sink_handle', 
        shape=(512, 512, 3),
        group='handle',
    ),
    6: dict(
        name='sink_top', 
        shape=(512,512,3),
        group='surface',
    ),
    7: dict(
        name='drawer', 
        shape=(512,512,3),
        group='surface',
    ),
    10: dict(
        name='sdoor_handle', 
        shape=(512,512,3),
        group='handle',
    ),
    11: dict(
        name='sdoor_surface', 
        shape=(512,512,3),
        group='surface',
    ),
    12: dict(
        name='lrdoor_surface', 
        shape=(512,512,3),
        group='surface',
    ),
    13: dict(
        name='lrdoor_handle', 
        shape=(512, 512, 3),
        group='handle',
    ),
    14: dict(
        name='micro_handle', 
        shape=(512, 512, 3),
        group='handle',
    ),
    
    16: dict(
        name='kettle_handle', 
        shape=(512, 512, 3),
        group='handle',
    ),
}
TEX_DIR = '/Users/mandizhao/mj_envs/mj_envs/envs/multi_task/common/kitchen/'
DEFAULT_TEXTURE_KWARGS = {
    'tex_ids': [1, 5, 6, 7, 10, 11, 12, 13, 14, 16],
    'tex_names': {
        'surface': [
            'wood',
            'stone',
            ],
        'handle': [
            'metal',
            ],
        'floor': [
            'tile'
            ],
        },
    'tex_path':  TEX_DIR + "/textures/*/*.png",
    }
OBJ_JNT_RANGE = {
    'lightswitch_joint': (-0.6, 0), 
    'rightdoorhinge': (0, 1), 
    'slidedoor_joint': (0, 0.3), 
    'leftdoorhinge': (-1, 0), 
    'micro0joint': (-1, 0), 
    'knob1_joint': (-1, 0), 
    'knob2_joint': (-1, 0),   
    'knob3_joint': (-1, 0), 
    'knob4_joint': (-1, 0), 
}
OBJ_LIST = list(OBJ_JNT_RANGE.keys())

DEFAULT_BODY_RANGE = {
            # "kitchen1": {
            #     "pos": {
            #         "center": [-0.1, 0.75, 0],
            #         "low":    [-.1, -.1, -.1],
            #         "high":   [.1, .1, .1],
            #         },
            #     "euler": {
            #         "center": [0, 0, 0],
            #         "low":    [0, 0, -.15],
            #         "high":   [0, 0, .15],
            #         },             
            # },
            "counters": { # note this includes both left counter and right sink
                "pos": {
                    "center": [0, 0, 0],
                    "low":    [0, -.4, 0],
                    "high":   [0, .4, 0],
                },
                "euler": {
                    "center": [0, 0, 0],
                    "low":    [0, 0, 0],
                    "high":   [0, 0, 0],
                }
            },
            "microwave": {
                "pos": {
                    "center": [-0.750, -0.025, 1.6],
                    "low":    [-.1, -.07, 0],
                    "high":   [0.05, 0.075,0],            
                    },
                "euler": {
                    "center": [0, 0, 0.3],
                    "low":    [0, 0, -.15],
                    "high":   [0, 0, .15],
                },
            },
            "hingecabinet": {
                "pos": {
                    "center": [-0.504, 0.28, 2.6],
                    "low":    [-.1, -.1, 0],
                    "high":   [0, .05, .1],
                },
                "euler": {
                    "center": [0, 0, 0],
                    "low":    [0, 0, 0],
                    "high":   [0, 0, 0],
                },
            },
            "slidecabinet": {
                "pos": {
                    "center":  [0.4, 0.28, 2.6], #None,  # use hingecabinet randomzied pos
                    "low":     [0, 0, 0],
                    "high":    [0.1, 0, 0],
                },
                "euler": {
                    "center": [0, 0, 0],
                    "low":    [0, 0, 0],
                    "high":   [0, 0, 0],
                },
            },
            "kettle0": {
                "pos": {
                    "center":  [-0.269, 0.35, 1.626],  
                    "low":     [0, 0, 0],
                    "high":    [0.5, 0.45, 0],
                },
                "euler": {
                    "center": [0, 0, 0],
                    "low":    [0, 0, 0],
                    "high":   [0, 0, 0],
                },
            }
        }


class KitchenFrankaAugment(KitchenFrankaFixed):

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)
        super()._setup(**kwargs)

        # set default kwargs for randomization
        self.augment_types = []
        self.body_rand_kwargs = DEFAULT_BODY_RANGE
        self.texture_modder = TextureModder(self.sim)
        self.texture_rand_kwargs = DEFAULT_TEXTURE_KWARGS 
        self.original_obj_goals = deepcopy(self.obj_goal) # save original obj_goal!
        self.joints_rand_kwargs = {
            'num_objects': 7,
            'non_target_objects': [obj for obj in OBJ_LIST if obj not in self.input_obj_goal.keys()]
        }
        

        self.light_rand_kwargs = {
            'ambient': {
                'low': -0.1,
                'high': 0.2,
                'center': deepcopy(self.sim.model.light_ambient), 
            },
            'diffuse': {
                'low': -0.3,
                'high': 0.3,
                'center': deepcopy(self.sim.model.light_diffuse),
            },
            'specular': {
                'low': -10,
                'high': 10,
                'center': deepcopy(self.sim.model.light_specular),
            }
        }

        self.goal = None 


    def set_augment_kwargs(self, aug_kwargs):
        self.augment_types = aug_kwargs.get('augment_types', [])

        # override default kwargs
        self.body_rand_kwargs.update(
            aug_kwargs.get('body', {})) 
        
        self.texture_rand_kwargs.update(
            aug_kwargs.get('texture', {}))
        if 'texture' in self.augment_types:
            texture_files = glob(self.texture_rand_kwargs['tex_path'])
            assert len(texture_files) > 0, "No texture files found at path: {}".format(self.texture_rand_kwargs['tex_path'])
        
        self.joints_rand_kwargs.update(
            aug_kwargs.get('joint', {}))

        self.light_rand_kwargs.update(
            aug_kwargs.get('light', {}))

    def get_augment_kwargs(self):
        return {
            'augment_types': self.augment_types,
            'body': self.body_rand_kwargs,
            'texture': self.texture_rand_kwargs,
            'joint': self.joints_rand_kwargs,
            'light': self.light_rand_kwargs,
        }

    def randomize_body_pose(self):
        def body_rand(name):
            kwargs = self.body_rand_kwargs.get(name, None)
            assert kwargs is not None, "body {} not found in body_rand_kwargs".format(name)
            pos = np.array(kwargs['pos']['center']) + \
                self.np_random.uniform(low=kwargs['pos']['low'], high=kwargs['pos']['high'])

            euler = np.array(kwargs['euler']['center']) + \
                self.np_random.uniform(low=kwargs['euler']['low'], high=kwargs['euler']['high'])
            
            bid = self.sim.model.body_name2id(name)
            self.sim.model.body_pos[bid] = pos
            self.sim.model.body_quat[bid] = euler2quat(euler)
            return pos, euler

        # dk_pos, _ = body_rand('desk')
        if 'micro0joint' not in self.input_obj_goal.keys():
            body_rand('microwave')
        #hc_pos, _  = body_rand('hingecabinet')
        body_rand('counters')

        if 'leftdoorhinge' not in self.input_obj_goal.keys() and \
            'rightdoorhinge' not in self.input_obj_goal.keys():
            body_rand('hingecabinet')
        # self.body_rand_kwargs['slidecabinet']['pos']['center'] = hc_pos
        if 'slidedoor_joint' not in self.input_obj_goal.keys():
            body_rand('slidecabinet') 
        # self.body_rand_kwargs['kettle0']['pos']['center'] = dk_pos
        body_rand('kettle0')

    def randomize_texture(self):
        def set_bitmap(tex_id, modder, new_bitmap):
            texture = modder.textures[tex_id]
            curr_bitmap = texture.bitmap
            assert curr_bitmap.dtype == new_bitmap.dtype and curr_bitmap.shape == new_bitmap.shape, \
                 f'Texture ID: {tex_id}: Incoming bitmap shape {new_bitmap.shape} and dtype {new_bitmap.dtype} does not match current bitmap: {curr_bitmap.shape}, {curr_bitmap.dtype}'
            modder.textures[tex_id].bitmap[:] = new_bitmap
            
            if not modder.sim.render_contexts:
                cymj.MjRenderContextOffscreen(modder.sim)
            for render_context in modder.sim.render_contexts:
                render_context.upload_texture(texture.id)
            return 

        tex_ids = self.texture_rand_kwargs.get('tex_ids', [])
        tex_files = glob(self.texture_rand_kwargs['tex_path'])
        assert len(tex_files) > 0, "No texture files found"
        for tex_id in tex_ids:
            tex_info = TEXTURE_ID_TO_INFOS.get(tex_id, None)
            assert tex_info is not None, f'ID {tex_id} not found'
            texture_keys = self.texture_rand_kwargs['tex_names'].get(tex_info['group'], None)
            assert texture_keys is not None, f"Texture group {tex_info['group']} not found"
            found_files = [f for f in tex_files if any([t in f for t in texture_keys])]

            fname = random.choice(found_files)
            new_tex = PIL.Image.open(fname).convert('RGB')

            if np.asarray(new_tex).shape != tex_info['shape']:
                new_tex = new_tex.resize(
                    (tex_info['shape'][0], tex_info['shape'][1])
                    )

            new_tex = np.asarray(new_tex, dtype=np.uint8)
            set_bitmap(tex_id, self.texture_modder, new_tex)
        return 

    def randomize_object_joints(self):
        object_keys = self.joints_rand_kwargs.get('non_target_objects', [])
        num_objects = self.joints_rand_kwargs.get('num_objects', 0)
        assert len(object_keys) > 0, "No non-target objects found"
        side_objs = random.sample(object_keys, num_objects)
        new_vals = []
        for side_obj_name in side_objs:
            val_range = OBJ_JNT_RANGE[side_obj_name]
            dof_adr = self.obj[side_obj_name]["dof_adr"] 
            new_val = np.random.uniform(low=val_range[0], high=val_range[1])
            new_vals.append( (side_obj_name, dof_adr, new_val) )

        env_state = self.get_env_state()
        new_obj_goal = deepcopy(self.original_obj_goals)
        for (side_obj_name, dof_adr, new_val) in new_vals: 
            env_state['qpos'][dof_adr] = new_val
            # NOTE: need to also reset the goal joint for each randomized side object
            goal_adr = self.obj[side_obj_name]["goal_adr"]
            # print(f'{side_obj_name}: old: {new_obj_goal[goal_adr]}, new goal {new_val}')
            new_obj_goal[goal_adr] = new_val
            
        
        self.set_obj_goal(obj_goal=new_obj_goal)

        self.set_state(
                qpos=env_state['qpos'], 
                qvel=env_state['qvel']
                )

        self.set_sim_obsd_state(
                qpos=env_state['qpos'],
                qvel=env_state['qvel']
                )
        return 

    def randomize_lights(self):
        for i in range(4):
            if 'ambient' in self.light_rand_kwargs:
                low = self.light_rand_kwargs['ambient']['low']
                high = self.light_rand_kwargs['ambient']['high']
                center = self.light_rand_kwargs['ambient']['center']
                new_vals = np.random.uniform(low, high, size=1)
                self.sim.model.light_ambient[i, :] = center[i] + new_vals
        
            if 'diffuse' in self.light_rand_kwargs:
                low = self.light_rand_kwargs['diffuse']['low']
                high = self.light_rand_kwargs['diffuse']['high']
                center = self.light_rand_kwargs['diffuse']['center']
                new_vals = np.random.uniform(low, high, size=1)
                self.sim.model.light_diffuse[i, :] = center[i] + new_vals
            
            if 'specular' in self.light_rand_kwargs:
                low = self.light_rand_kwargs['specular']['low']
                high = self.light_rand_kwargs['specular']['high']
                center = self.light_rand_kwargs['specular']['center']
                new_vals = np.random.uniform(low, high, size=1)
                self.sim.model.light_specular[i, :] = center[i] + new_vals

    def reset(self, reset_qpos=None, reset_qvel=None):
        # random reset of robot initial pos 

        if reset_qpos is None:
            reset_qpos = self.init_qpos.copy()
            reset_qpos[self.robot_dofs] += (
                0.05
                * (self.np_random.uniform(size=len(self.robot_dofs)) - 0.5)
                * (self.robot_ranges[:, 1] - self.robot_ranges[:, 0])
            )
        super().reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel)

        if 'body' in self.augment_types:
            self.randomize_body_pose()

        if 'texture' in self.augment_types:
            self.randomize_texture()
        
        if 'joint' in self.augment_types:
            self.randomize_object_joints()

        if 'light' in self.augment_types:
            self.randomize_lights()
        
        return self.get_obs()

    def set_sim_obsd_state(self, qpos=None, qvel=None, act=None):
        """
        Set MuJoCo sim_obsd state
        """
        sim = self.sim_obsd
        assert qpos.shape == (sim.model.nq,) and qvel.shape == (sim.model.nv,)
        old_state = sim.get_state()
        if qpos is None:
            qpos = old_state.qpos
        if qvel is None:
            qvel = old_state.qvel
        if act is None:
            act = old_state.act
        new_state = mujoco_py.MjSimState(old_state.time, qpos=qpos, qvel=qvel, act=act, udd_state={})
        sim.set_state(new_state)
        sim.forward()

    def make_copy_env(self):
        copy_env = deepcopy(self)

        # match texture
        for tex_id in self.texture_rand_kwargs.get('tex_ids', []):
            copy_env.texture_modder.textures[tex_id].bitmap[:] = self.texture_modder.textures[tex_id].bitmap
        
        # match qpos and qvel
        qpos, qvel = self.get_env_state()['qpos'], self.get_env_state()['qvel']
        copy_env.set_state(qpos=qpos, qvel=qvel)
        copy_env.set_sim_obsd_state(qpos=qpos, qvel=qvel)
        copy_env.set_obj_goal(self.obj_goal)
        
        # match lightings
        copy_env.sim.model.light_specular[:] = self.sim.model.light_specular
        copy_env.sim.model.light_diffuse[:] = self.sim.model.light_diffuse
        copy_env.sim.model.light_ambient[:] = self.sim.model.light_ambient

        # match body pose
        copy_env.sim.model.body_pos[:] = self.sim.model.body_pos
        copy_env.sim.model.body_quat[:] = self.sim.model.body_quat

        return copy_env

    def set_goal(
        self, 
        expert_traj, 
        cameras=['left_cam', 'right_cam'], 
        goal_window=5, 
        frame_size=(256, 256),
        min_success_count=5,
        device_id=0,
        max_trials=10,
        verbose=False,
        ):
        """
        Use for online evaluation, such that the goal image gets applied the sample randomization.
        Call env.set_goal(**kwargs) after every self.reset(). Note that this method calls another reset() to the randomization 
        Input takes in a single expert trajectory, replay the actions to render a goal image
        """
        assert type(expert_traj) is dict, "Expert trajectory must be a dictionary"
        expert_actions = expert_traj['actions']
        horizon = expert_actions.shape[0]
        goal_tstep = np.random.randint(low=horizon-goal_window, high=horizon)
        new_camera_imgs = None 
        success_count = 0 
        goal_set = False 

        
        init_state = {key: v[0] for key, v in expert_traj['env_states'].items()}
        self.reset(reset_qpos=init_state['qpos'], reset_qvel=init_state['qvel'])
  
        for trial in range(max_trials):
            goal_env = self.make_copy_env()
            for t, action in enumerate(expert_actions): 
                next_o, rwd, done, next_env_info = goal_env.step(action)
                if next_env_info.get('solved', False):
                    success_count += 1
                if t == goal_tstep and success_count >= min_success_count:
                    goal_set = True
                    curr_frame = goal_env.render_camera_offscreen(
                        sim=goal_env.sim,
                        cameras=cameras,
                        width=frame_size[0],
                        height=frame_size[1],
                        device_id=device_id
                        ) 
                    new_camera_imgs = {cam: curr_frame[i] for i, cam in enumerate(cameras)}
            if verbose:
                print("Trial {}: {}".format(trial, success_count))
            del goal_env
            if goal_set:
                break

        if not goal_set:
            raise ValueError("Failed to complete the task by replaying the expert actions")
        self.goal_imgs = new_camera_imgs
            
        return new_camera_imgs