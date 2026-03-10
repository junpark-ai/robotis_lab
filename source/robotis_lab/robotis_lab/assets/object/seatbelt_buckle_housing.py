import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from robotis_lab.assets.object import ROBOTIS_LAB_OBJECT_ASSETS_DATA_DIR


SEATBELT_BUCKLE_HOUSING_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOTIS_LAB_OBJECT_ASSETS_DATA_DIR}/object/seatbelt_buckle_housing.usd",
        scale=(0.001, 0.001, 0.001),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=[0.0, 0.0, 0.0],
        rot=[1.0, 0.0, 0.0, 0.0],
    ),
)
