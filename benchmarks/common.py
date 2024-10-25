import numpy as np
import pinocchio as pin
from tap import Tap
from typing import Literal


class Args(Tap):
    solver: Literal["ali", "ipopt", "altro"]
    record: bool = False
    viz: bool = False


def load_solo12(verbose=False):
    """Load Solo12 with Euclidean parameterization."""
    from os.path import join
    from example_robot_data.robots_loader import (
        Solo12Loader,
        RobotWrapper,
        getModelPath,
        readParamsFromSrdf,
    )

    jmc = pin.JointModelComposite(pin.JointModelTranslation())
    jmc.addJoint(pin.JointModelSphericalZYX())
    df_path = join(
        Solo12Loader.path, Solo12Loader.urdf_subpath, Solo12Loader.urdf_filename
    )
    model_path = getModelPath(df_path, verbose)
    df_path = join(model_path, df_path)
    builder = RobotWrapper.BuildFromURDF
    robot = builder(df_path, [join(model_path, "../..")], jmc)
    srdf_path = join(
        model_path,
        Solo12Loader.path,
        Solo12Loader.srdf_subpath,
        Solo12Loader.srdf_filename,
    )
    robot.q0 = readParamsFromSrdf(
        robot.model,
        srdf_path,
        verbose,
        Solo12Loader.has_rotor_parameters,
        Solo12Loader.ref_posture,
    )

    return robot


def add_obj_viz(visual_model, name, pos):
    import hppfcl as coal

    pose = pin.SE3.Identity()
    obj_color = np.array([134, 10, 50, 200]) / 255.0
    pose.translation[:] = pos

    gobj = pin.GeometryObject(name, 0, coal.Sphere(0.05), pose)
    gobj.meshColor[:] = obj_color
    return visual_model.addGeometryObject(gobj)
