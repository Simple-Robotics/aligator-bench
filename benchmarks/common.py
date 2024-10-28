import numpy as np
import pinocchio as pin
from tap import Tap
from typing import Literal


class Args(Tap):
    solver: Literal["ali", "ipopt", "altro"]
    record: bool = False
    viz: bool = False


def load_robot_euclidean(loader_cls, verbose=False):
    """Load a robot with Euclidean parameterization."""
    from os.path import join
    from example_robot_data.robots_loader import (
        RobotWrapper,
        getModelPath,
        readParamsFromSrdf,
    )

    jmc = pin.JointModelComposite(pin.JointModelTranslation())
    jmc.addJoint(pin.JointModelSphericalZYX())
    df_path = join(loader_cls.path, loader_cls.urdf_subpath, loader_cls.urdf_filename)
    model_path = getModelPath(df_path, verbose)
    df_path = join(model_path, df_path)
    builder = RobotWrapper.BuildFromURDF
    robot = builder(df_path, [join(model_path, "../..")], jmc)
    if len(loader_cls.srdf_filename) > 0:
        srdf_path = join(
            model_path,
            loader_cls.path,
            loader_cls.srdf_subpath,
            loader_cls.srdf_filename,
        )
        robot.q0 = readParamsFromSrdf(
            robot.model,
            srdf_path,
            verbose,
            loader_cls.has_rotor_parameters,
            loader_cls.ref_posture,
        )

    return robot


def load_solo12(verbose=False):
    """Load Solo12 with Euclidean parameterization."""
    from example_robot_data.robots_loader import Solo12Loader

    return load_robot_euclidean(Solo12Loader, verbose)


def load_hector(verbose=False):
    from example_robot_data.robots_loader import HectorLoader

    return load_robot_euclidean(HectorLoader, verbose)


def add_obj_viz(visual_model, name, pos):
    import hppfcl as coal

    pose = pin.SE3.Identity()
    obj_color = np.array([134, 10, 50, 200]) / 255.0
    pose.translation[:] = pos

    gobj = pin.GeometryObject(name, 0, coal.Sphere(0.05), pose)
    gobj.meshColor[:] = obj_color
    return visual_model.addGeometryObject(gobj)
