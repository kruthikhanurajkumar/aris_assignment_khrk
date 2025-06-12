import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# Robot Parameters
l1 = l2 = 1.0
Kp = 25
dt = 0.01

# Joint Angles
theta_base = 0
theta1 = np.pi / 4
theta2 = np.pi / 4

# Target Point
target = np.array([1.0, 0.0, 0.5])

# History of wrist
history = [[], [], []]

# Global for plot elements
fig, ax = None, None
arm_line = None
path_line = None
target_scatter = None


def on_click(event):
    if not event.inaxes:
        return
    global target
    x = input("Enter X coordinate: ")
    y = input("Enter Y coordinate: ")
    z = input("Enter Z coordinate: ")
    try:
        x = float(x)
        y = float(y)
        z = float(z)
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return
    if not (-2 <= x <= 2 and -2 <= y <= 2 and 0 <= z <= 2):
        print("Coordinates out of bounds. Please enter values within the range.")
        return
    target = np.array([x, y, z])
    run(target)


def on_key(event):
    global history, path_line
    if event.key in ['e', 'E']:
        plt.close('all')
        sys.exit(0)
    elif event.key in ['c', 'C']:
        history = [[], [], []]
        path_line.set_data([], [])
        path_line.set_3d_properties([])


def cartesian_to_cylindrical(x, y, z):
    r = np.hypot(x, y)
    theta_base = np.arctan2(y, x)
    return r, z, theta_base


def inverse_kinematics_rz(r, z):
    try:
        d = np.hypot(r, z)
        if d > l1 + l2:
            raise ValueError("Target unreachable")

        cos_theta2 = (d ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
        theta2 = np.arccos(np.clip(cos_theta2, -1, 1))

        k1 = l1 + l2 * np.cos(theta2)
        k2 = l2 * np.sin(theta2)
        theta1 = np.arctan2(z, r) - np.arctan2(k2, k1)

        return True, theta1, theta2
    except Exception as e:
        print("IK failed:", e)
        return False, 0, 0


def forward_kinematics(theta_base, theta1, theta2):
    elbow_rz = np.array([
        l1 * np.cos(theta1),
        l1 * np.sin(theta1)
    ])
    wrist_rz = elbow_rz + np.array([
        l2 * np.cos(theta1 + theta2),
        l2 * np.sin(theta1 + theta2)
    ])

    def rz_to_xyz(r, z):
        x = r * np.cos(theta_base)
        y = r * np.sin(theta_base)
        return np.array([x, y, z])

    shoulder = np.array([0, 0, 0])
    elbow = rz_to_xyz(elbow_rz[0], elbow_rz[1])
    wrist = rz_to_xyz(wrist_rz[0], wrist_rz[1])

    return shoulder, elbow, wrist


def run(target_point):
    global theta1, theta2, theta_base

    x, y, z = target_point
    r, z_cyl, theta_base = cartesian_to_cylindrical(x, y, z)
    ok, goal_theta1, goal_theta2 = inverse_kinematics_rz(r, z_cyl)

    if not ok:
        return

    for _ in range(300):
        theta1 += Kp * (goal_theta1 - theta1) * dt
        theta2 += Kp * (goal_theta2 - theta2) * dt
        plot(theta_base, theta1, theta2)


def plot(theta_base, theta1, theta2):
    global history, arm_line, path_line, target_scatter

    shoulder, elbow, wrist = forward_kinematics(theta_base, theta1, theta2)

    history[0].append(wrist[0])
    history[1].append(wrist[1])
    history[2].append(wrist[2])
    
    arm_line.set_data([shoulder[0], elbow[0], wrist[0]],
                      [shoulder[1], elbow[1], wrist[1]])
    arm_line.set_3d_properties([shoulder[2], elbow[2], wrist[2]])


    path_line.set_data(history[0], history[1])
    path_line.set_3d_properties(history[2])


    target_scatter.remove()
    target_scatter = ax.scatter(*target, c='r', s=40, label='Target')

    plt.pause(dt)


def main():
    global fig, ax, arm_line, path_line, target_scatter

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([0, 2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3-DOF Cylindrical Arm IK Simulation")
    ax.view_init(elev=30, azim=135)

    # Initial empty plots
    arm_line, = ax.plot([0, 0], [0, 0], [0, 0], 'ko-', linewidth=2, markersize=6)
    path_line, = ax.plot([], [], [], 'b--', label='Path')
    target_scatter = ax.scatter(*target, c='r', s=40, label='Target')

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    try:
        run(target)
        plt.legend()
        plt.show()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Closing...")
        plt.close('all')
        sys.exit(0)


if __name__ == "__main__":
    main()
