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
        shape=(512,512,3),
        group='handle',
    ),
    14: dict(
        name='micro_handle', 
        shape=(1024,1024,3),
        group='handle',
    ),
    
    16: dict(
        name='kettle_handle', 
        shape=(512,512,3),
        group='handle',
    ),
}
TEX_DIR = '/Users/mandizhao/mj_envs/mj_envs/envs/relay_kitchen'
OBJ_JNT_RANGE = {
    'lightswitch_joint': (-0.7, 0), 
    'rightdoorhinge': (0, 1.57), 
    'slidedoor_joint': (0, 0.44), 
    'leftdoorhinge': (-1.25, 0), 
    'micro0joint': (-1.25, 0), 
    'knob1_joint': (0, -1.57), 
    'knob2_joint': (0, -1.57),   
    'knob3_joint': (-1.57, 0), 
    'knob4_joint': (-1.57, 0), 
}
OBJ_LIST = list(OBJ_JNT_RANGE.keys())

class KitchenFrankaAugment(KitchenFrankaFixed):

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)
        super()._setup(**kwargs)

        self.augment_types = []
        self.body_rand_kwargs = {
            "desk": {
                "pos": {
                    "center": [-0.1, 0.75, 0],
                    "low":    [-.1, -.1, -.1],
                    "high":   [.1, .1, .1],
                    },
                "euler": {
                    "center": [0, 0, 0],
                    "low":    [0, 0, -.15],
                    "high":   [0, 0, .15],
                    },             
            },
            "microwave": {
                "pos": {
                    "center": [-0.750, -0.025, 1.6],
                    "low":    [-.1, -.1, 0],
                    "high":   [.1, .1, .1],            
                    },
                "euler": {
                    "center": [0, 0, .785],
                    "low":    [0, 0, -.15],
                    "high":   [0, 0, .15],
                },
            },
            "hingecabinet": {
                "pos": {
                    "center": [-0.504, 0.28, 2.6],
                    "low":    [-.1, -.1, 0],
                    "high":   [.1, .1, .1],
                },
                "euler": {
                    "center": [0, 0, 0],
                    "low":    [0, 0, 0],
                    "high":   [0, 0, 0],
                },
            },
            "slidecabinet": {
                "pos": {
                    "center":  None,  # use hingecabinet randomzied pos
                    "low":     [.904, 0, 0],
                    "high":    [1.1, 0, 0],
                },
                "euler": {
                    "center": [0, 0, 0],
                    "low":    [0, 0, 0],
                    "high":   [0, 0, 0],
                },
            },
            "kettle0": {
                "pos": {
                    "center":  None, # use desk randomzied pos
                    "low":     [-.2, -.3, 1.626],
                    "high":    [.4, 0.3, 1.626]
                },
                "euler": {
                    "center": [0, 0, 0],
                    "low":    [0, 0, 0],
                    "high":   [0, 0, 0],
                },
            }
        }

        self.texture_modder = TextureModder(self.sim)
        self.texture_rand_kwargs = {
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
            'tex_path':  TEX_DIR + "/assets/textures/*/*.png",
            }
        self.joints_rand_kwargs = {
            'num_objects': 7,
            'non_target_objects': [obj for obj in OBJ_LIST if obj not in self.input_obj_goal.keys()]
        }

    def set_augment_kwargs(self, aug_kwargs):
        self.augment_types = aug_kwargs.get('augment_types', [])

        body_rand_kwargs = aug_kwargs.get('body', None) 
        if body_rand_kwargs is not None:
            self.body_rand_kwargs.update(body_rand_kwargs)
        
        texture_rand_kwargs = aug_kwargs.get('texture', None)
        if texture_rand_kwargs is not None:
            self.texture_rand_kwargs.update(texture_rand_kwargs)
        
        joints_rand_kwargs = aug_kwargs.get('joints', None)
        if joints_rand_kwargs is not None:
            self.joints_rand_kwargs.update(joints_rand_kwargs)

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
        body_rand('microwave')
        hc_pos, _  = body_rand('hingecabinet')

        self.body_rand_kwargs['slidecabinet']['pos']['center'] = hc_pos
        body_rand('slidecabinet')

        # self.body_rand_kwargs['kettle0']['pos']['center'] = dk_pos
        # body_rand('kettle0')

    def randomize_texture(self):
        def set_bitmap(tex_id, modder, new_bitmap):
            texture = modder.textures[tex_id]
            curr_bitmap = texture.bitmap
            assert curr_bitmap.dtype == new_bitmap.dtype and curr_bitmap.shape == new_bitmap.shape, \
                 f'Incoming bitmap shape and dtype does not match current bitmap'
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
            found_files = [
                f for f in tex_files if any([t in f for t in texture_keys])
            ]

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
        for side_obj in side_objs:
            val_range = OBJ_JNT_RANGE[side_obj]
            dof_adr = self.obj[side_obj]["dof_adr"] 
            new_val = np.random.uniform(low=val_range[0], high=val_range[1])
            new_vals.append( (dof_adr, new_val) )

        env_state = self.get_env_state()
        for (dof_adr, new_val) in new_vals:
            env_state['qpos'][dof_adr] = new_val
        
        self.set_state(
                qpos=env_state['qpos'], 
                qvel=env_state['qvel']
                )
        return 

        

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

        # if 'body' in self.augment_types:
        #     self.randomize_body_pose()
        if 'texture' in self.augment_types:
            self.randomize_texture()
        
        if 'joint' in self.augment_types:
            self.randomize_object_joints()

        
        return self.get_obs()

