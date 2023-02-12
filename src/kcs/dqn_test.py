import numpy as np
import test_wp_track as wpt

model_name = 'model_001'

def ellipse(flag=0):
    npoints = 12

    if flag == 0:
        # Starboard turning ellipse
        theta = np.linspace(0, 2 * np.pi, num=npoints, endpoint=True)
        theta_des = np.linspace(0, 2 * np.pi, num=1000)
        psi0 = np.pi / 2
    else:
        # Port turning ellipse
        theta = np.linspace(0, -2 * np.pi, num=npoints, endpoint=True)
        theta_des = np.linspace(0, -2 * np.pi, num=1000)
        psi0 = -np.pi / 2

    a_ellipse = 12
    b_ellipse = 12
    x_wp = a_ellipse * np.cos(theta)
    y_wp = b_ellipse * np.sin(theta)

    xdes = a_ellipse * np.cos(theta_des)
    ydes = b_ellipse * np.sin(theta_des)

    return npoints, x_wp, y_wp, psi0, xdes, ydes

def straight():
    npoints = 9
    psi0 = 0
    x_wp = np.linspace(0, 64, num=npoints, endpoint=True)
    y_wp = 0 * x_wp
    xdes = x_wp
    ydes = y_wp
    return npoints, x_wp, y_wp, psi0, xdes, ydes

def eight(flag=0):
    npoints = 15
    psi0 = 0
    circ_dia = 9

    if flag == 0:
        theta1 = np.linspace(-0.5 * np.pi, 1.5 * np.pi, npoints - npoints // 2, endpoint=True)
        theta1_des = np.linspace(-0.5 * np.pi, 1.5 * np.pi, 1000)
        x1 = circ_dia * np.cos(theta1)
        y1 = circ_dia * np.sin(theta1) + circ_dia
        x1des = circ_dia * np.cos(theta1_des)
        y1des = circ_dia * np.sin(theta1_des) + circ_dia

        theta2 = np.linspace(0.5 * np.pi, 2.5 * np.pi, npoints // 2, endpoint=False)
        theta2_des = np.linspace(0.5 * np.pi, 2.5 * np.pi, 1000)
        x2 = circ_dia * np.cos(theta2)
        y2 = circ_dia * np.sin(theta2) - circ_dia
        x2des = circ_dia * np.cos(theta2_des)
        y2des = circ_dia * np.sin(theta2_des) - circ_dia

        x_wp = np.append(x1, x2[::-1])
        y_wp = np.append(y1, y2[::-1])

        xdes = np.append(x1des, x2des[::-1])
        ydes = np.append(y1des, y2des[::-1])

    else:
        theta1 = np.linspace(-0.5 * np.pi, 1.5 * np.pi, npoints - npoints // 2, endpoint=True)
        theta1_des = np.linspace(-0.5 * np.pi, 1.5 * np.pi, 1000)
        x1 = circ_dia * np.cos(theta1)
        y1 = circ_dia * np.sin(theta1) + circ_dia
        x1des = circ_dia * np.cos(theta1_des)
        y1des = circ_dia * np.sin(theta1_des) + circ_dia

        theta2 = np.linspace(2.5 * np.pi, 0.5 * np.pi, npoints // 2, endpoint=False)
        theta2_des = np.linspace(2.5 * np.pi, 0.5 * np.pi, 1000)
        x2 = circ_dia * np.cos(theta2)
        y2 = circ_dia * np.sin(theta2) - circ_dia
        x2des = circ_dia * np.cos(theta2_des)
        y2des = circ_dia * np.sin(theta2_des) - circ_dia

        x_wp = np.append(x2, x1)
        y_wp = np.append(y2, y1)
        xdes = np.append(x2des, x1des[::-1])
        ydes = np.append(y2des, y1des[::-1])
        
    return npoints, x_wp, y_wp, psi0, xdes, ydes

def single_wp(quadrant=1, len=10):
    npoints = 2
    psi0 = 0

    if quadrant == 1:
        x_wp = np.array([0, len])
        y_wp = np.array([0, len])
    elif quadrant == 2:
        x_wp = np.array([0, -len])
        y_wp = np.array([0, len])
    elif quadrant == 3:
        x_wp = np.array([0, -len])
        y_wp = np.array([0, -len])
    elif quadrant == 4:
        x_wp = np.array([0, len])
        y_wp = np.array([0, -len])

    xdes = x_wp
    ydes = y_wp
    return npoints, x_wp, y_wp, psi0, xdes, ydes

# No wind condition

# Single waypoint tracking in four quadrants
# n, xwp, ywp, psi0, xdes, ydes = single_wp(quadrant=1)
# wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='quadrant_01', xdes=xdes, ydes=ydes)
# n, xwp, ywp, psi0, xdes, ydes = single_wp(quadrant=2)
# wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='quadrant_02', xdes=xdes, ydes=ydes)
# n, xwp, ywp, psi0, xdes, ydes = single_wp(quadrant=3)
# wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='quadrant_03', xdes=xdes, ydes=ydes)
# n, xwp, ywp, psi0, xdes, ydes = single_wp(quadrant=4)
# wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='quadrant_04', xdes=xdes, ydes=ydes)

# Ellipse starboard turn
# n, xwp, ywp, psi0, xdes, ydes = ellipse(flag=0)
# wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='ellipse_stbd', xdes=xdes, ydes=ydes)

# Ellipse port turn
n, xwp, ywp, psi0, xdes, ydes = ellipse(flag=1)
wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='ellipse_port', xdes=xdes, ydes=ydes)

# Straight line
# n, xwp, ywp, psi0, xdes, ydes = straight()
# wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='straight', xdes=xdes, ydes=ydes)

# # Eight bottom first
# n, xwp, ywp, psi0, xdes, ydes = eight(flag=0)
# wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='eight_bottom', xdes=xdes, ydes=ydes)

# # Eight top first
# n, xwp, ywp, psi0, xdes, ydes = eight(flag=1)
# wpt.wp_track(model_name, wind_flag=0, wind_speed=0, wind_dir=0, npoints=n, x_wp=xwp, y_wp=ywp, psi0=psi0, traj_str='eight_top', xdes=xdes, ydes=ydes)
