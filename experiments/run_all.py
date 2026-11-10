
"""
Launch all 80 MNIST experiments.

Each experiment configuration is trained using five random seeds.
"""

print("Starting all MNIST experiments...")

print("\n[1/8] Baseline")
import experiments.baseline

print("\n[2/8] Rotation")
import experiments.rotation

print("\n[3/8] Translation")
import experiments.translation

print("\n[4/8] Cropping")
import experiments.cropping

print("\n[5/8] Flipping")
import experiments.flipping

print("\n[6/8] Elastic deformation")
import experiments.elastic_deformation

print("\n[7/8] Gaussian noise")
import experiments.gaussian_noise

print("\n[8/8] Strength ablation")
import experiments.strength_ablation

print("\nAll 80 experiment runs have finished!")
