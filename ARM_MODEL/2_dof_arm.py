import numpy as np
import matplotlib.pyplot as plt
import time
import sys

plt.ion()
Kp = 30
dt = 0.01
l1 = l2  = 1
xi = 1.0
yi = 1.0
theta1, theta2 = np.pi/2, np.pi

history = [[], []]
def onclick(event):
    global xi, yi
    xi , yi = event.xdata ,event.ydata
    run(xi, yi)

def onpress(event):
    global history
    if event.key == 'e' or event.key == 'E':
        plt.close()
        exit(0)
    elif event.key == 'c' or event.key == 'C':
        history = [[], []]

def inverse_kinematics(x, y):
    try:
        e= 0.00001
        theta2 = np.arccos((x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2))
        theta1 = np.arctan(y/abs(x + e)) + np.arcsin(l2 * np.sin(np.pi - theta2) / np.sqrt(x**2 + y**2 + e))
        if x < 0:
            theta1 = np.pi - (np.arctan(y/abs(x + e)) - np.arcsin(l2 * np.sin(np.pi - theta2) / np.sqrt(x**2 + y**2 + e)))
        return 1, theta1, theta2
    except:
        print('Goal Unreachable!')
        return 0, 0, 0
        sys.exit(1)
def run(xi, yi):
    global Kp, theta1, theta2
    check, goal_theta1, goal_theta2 = inverse_kinematics(xi, yi)
    if check:
        while True:
            theta1 += Kp * (goal_theta1 - theta1) * dt
            theta2 += Kp * (goal_theta2 - theta2) * dt
            plot(theta1, theta2, xi, yi)


def plot(theta1, theta2, xi, yi):
    global l1, l2, dt, history
    shoulder = np.array([0, 0])
    elbow = shoulder + l1 * np.array([np.cos(theta1), np.sin(theta1)])
    wrist = elbow + l2 * np.array([np.sin(np.pi/2 + theta2 - theta1), np.cos(np.pi/2 + theta2 - theta1)])

    history[0].append(wrist[0])
    history[1].append(wrist[1])

    domianx= np.linspace(-l1-l2, l1+l2, 100)
    domiany = np.sqrt((l1 + l2)**2 - domianx**2)
    plt.cla()
    plt.plot(domianx, domiany, 'r--', label='Reachable Area')

    plt.plot(shoulder[0], shoulder[1], 'ro', label='Shoulder')
    plt.plot(elbow[0], elbow[1], 'go', label='Elbow')
    # plt.plot(wrist[0], wrist[1], 'bo', label='Wrist')

    plt.plot([shoulder[0], elbow[0], wrist[0]], [shoulder[1], elbow[1], wrist[1]], 'k-', label='Arm')
    plt.plot(history[0], history[1], 'b-', label='Path')
    plt.xlim( -l1-l2, l1+l2 )
    plt.ylim( (-l1-l2)/4, l1+l2 ) 

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2-DOF Arm Simulation')
    plt.pause(dt)

def main():
    global xi , yi
    fig =plt.figure()
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onpress)
    run(xi, yi)

if __name__ == "__main__":
    main()
    
