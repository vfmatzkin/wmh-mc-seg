import os

from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection


def bias_field_correction(src_path, dst_path):
    print("N4ITK on: ", src_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    try:
        n4 = N4BiasFieldCorrection()
        n4.inputs.input_image = src_path
        n4.inputs.output_image = dst_path

        n4.inputs.dimension = 3
        n4.inputs.n_iterations = [100, 100, 60, 40]
        n4.inputs.shrink_factor = 3
        n4.inputs.convergence_threshold = 1e-4
        n4.inputs.bspline_fitting_distance = 300
        n4.run()
    except RuntimeError:
        print("\tFailed on: ", src_path)

    return
