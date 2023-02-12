import numpy as np

# Ship Geometry and Constants
L = 230
B = 32.2
d_em = 10.8
rho = 1025
g = 9.80665

Cb = 0.651
Dsp = Cb * L * B * d_em
Fn = 0.26
U_des = Fn * np.sqrt(g * L)
xG = -3.404
kzzp = 0.25

# Surge Hydrodynamic Derivatives in non-dimensional form
X0 = -0.0167
Xbb = -0.0549
Xbr_minus_my = -0.1084
Xrr = -0.0120
Xbbbb = -0.0417

# Sway Hydrodynamic Derivatives in non-dimensional form
Yb = 0.2252
Yr_minus_mx = 0.0398
Ybbb = 1.7179
Ybbr = -0.4832
Ybrr = 0.8341
Yrrr = -0.0050

# Yaw Hydrodynamic Derivatives in non-dimensional form
Nb = 0.1111
Nr = -0.0465
Nbbb = 0.1752
Nbbr = -0.6168
Nbrr = 0.0512
Nrrr = -0.0387

n_prop = 115.5 / 60
Dp = 7.9
wp = 1 - 0.645  # Effective Wake Fraction of the Propeller
tp = 1 - 0.793  # Thrust Deduction Factor

eps = 0.956
eta = 0.7979
kappa = 0.633
xp_P = -0.4565  # Assuming propeller location is 10 m ahead of AP (Rudder Location)
xp_R = -0.5

mp = Dsp / (0.5 * (L ** 2) * d_em)
xGp = xG / L
# Added Mass and Mass Moment of Inertia (from MDLHydroD)
mxp = 1790.85 / (0.5 * (L ** 2) * d_em)
myp = 44324.18 / (0.5 * (L ** 2) * d_em)
Jzzp = 140067300 / (0.5 * (L ** 4) * d_em)
Izzp = mp * (kzzp ** 2) + mp * (xGp ** 2)

a0 = 0.5228
a1 = -0.4390
a2 = -0.0609

tR = 1 - 0.742
aH = 0.361
xp_H = -0.436

A_R = L * d_em / 54.86
Lamda = 2.164
f_alp = 6.13 * Lamda / (2.25 + Lamda)


def KCS_ode(t, v, delta_c, wind_flag=0, wind_speed=0, wind_dir=0,
            wave_flag=0, wave_height=0, wave_period=0, wave_dir=0):

    # Nondimensional State Space Variables
    up = v[0]
    vp = v[1]
    rp = v[2]
    xp = v[3]
    yp = v[4]
    psi = v[5]
    delta = v[6]
    # n_prop = v[7]

    # Derived kinematic variables
    b = np.arctan2(-vp, up)  # Drift angle

    # ----------------------------------------------------
    # Hull Force Calculation
    # ----------------------------------------------------

    # Non-dimensional Surge Hull Hydrodynamic Force
    Xp_H = X0 * (up ** 2) \
           + Xbb * (b ** 2) + Xbr_minus_my * b * rp \
           + Xrr * (rp ** 2) + Xbbbb * (b ** 4)

    # Non-dimensional Sway Hull Hydrodynamic Force
    Yp_H = Yb * b + Yr_minus_mx * rp + Ybbb * (b ** 3) \
           + Ybbr * (b ** 2) * rp + Ybrr * b * (rp ** 2) \
           + Yrrr * (rp ** 3)

    # Non-dimensional Yaw Hull Hydrodynamic Moment
    Np_H = Nb * b + Nr * rp + Nbbb * (b ** 3) \
           + Nbbr * (b ** 2) * rp + Nbrr * b * (rp ** 2) \
           + Nrrr * (rp ** 3)

    # ----------------------------------------------------
    # Propulsion Force Calculation
    # ----------------------------------------------------
    # The value self propulsion RPM is taken from Yoshimura's SIMMAN study
    # Analysis of steady hydrodynamic force components and prediction of
    # manoeuvering ship motion with KVLCC1, KVLCC2 and KCS

    J = (up * U_des) * (1 - wp) / (n_prop * Dp)  # Advance Coefficient

    Kt = a0 + a1 * J + a2 * (J ** 2)  # Thrust Coefficient

    # Dimensional Propulsion Force
    X_P = (1 - tp) * rho * Kt * (Dp ** 4) * (n_prop ** 2)

    # Non-dimensional Propulsion Force
    Xp_P = X_P / (0.5 * rho * L * d_em * (U_des ** 2))

    # ----------------------------------------------------
    # Rudder Force Calculation
    # ----------------------------------------------------

    b_p = b - xp_P * rp

    if b_p > 0:
        gamma_R = 0.492
    else:
        gamma_R = 0.338

    lp_R = -0.755

    up_R = eps * (1 - wp) * up * np.sqrt(eta * (1 + kappa * (np.sqrt(1 + 8 * Kt / (np.pi * (J ** 2))) - 1)) ** 2 + (1 - eta))

    vp_R = gamma_R * (vp + rp * lp_R)

    Up_R = np.sqrt(up_R ** 2 + vp_R ** 2)
    alpha_R = delta - np.arctan2(-vp_R, up_R)

    F_N = A_R / (L * d_em) * f_alp * (Up_R ** 2) * np.sin(alpha_R)

    Xp_R = - (1 - tR) * F_N * np.sin(delta)
    Yp_R = - (1 + aH) * F_N * np.cos(delta)
    Np_R = - (xp_R + aH * xp_H) * F_N * np.cos(delta)

    # ----------------------------------------------------
    # Coriolis terms
    # ----------------------------------------------------

    mp = Dsp / (0.5 * (L ** 2) * d_em)
    xGp = xG / L

    Xp_C = mp * vp * rp + mp * xGp * (rp ** 2)
    Yp_C = -mp * up * rp
    Np_C = -mp * xGp * up * rp

    # ----------------------------------------------------
    # Wind Force Calculation
    # ----------------------------------------------------

    if wind_flag == 1:
        Vw = wind_speed  # wind speed
        betaw = wind_dir * (np.pi/180)  # wind direction
        Lp = 3.0464
        de = 0.1430
        uw = Vw * np.cos(betaw - psi)
        vw = Vw * np.sin(betaw - psi)
        urw = up - uw
        vrw = vp - vw
        Uwr = (urw ** 2 + vrw ** 2) ** 0.5
        gammaw = np.arctan2(-vrw, -urw)
        # print(gammaw,"gamma")

        rhow = 1025
        rhoa = 1.225
        Ax = (0.4265 * (0.2517 - 0.1430))
        Ay = ((0.2517 - 0.1430) * Lp)
        # print(Ax,Ay,"AX")

        Cwx = 1 * np.cos(gammaw)
        Cwy = 1 * np.sin(gammaw)
        Cwpsi = 0.5 * np.sin(gammaw)

        Xp_W = (Ax * Cwx * Uwr * abs(Uwr)) * rhoa / (Lp * de * rhow)
        Yp_W = (Ay * Cwy * Uwr * abs(Uwr)) * rhoa / (Lp * de * rhow)
        Np_W = (Ay * Lp * Cwpsi * Uwr * abs(Uwr)) * rhoa / ((Lp ** 2) * de * rhow)
    else:
        Xp_W = 0.0
        Yp_W = 0.0
        Np_W = 0.0

    # Net non-dimensional force and moment computation
    Xp = Xp_H + Xp_R + Xp_C + Xp_W + Xp_P
    Yp = Yp_H + Yp_R + Yp_C + Yp_W
    Np = Np_H + Np_R + Np_C + Np_W

    # Net force vector computation in Abkowitz non-dimensionalization
    X = Xp
    Y = Yp
    N = Np

    # Added Mass and Mass Moment of Inertia (from MDLHydroD)
    mxp = 1790.85 / (0.5 * (L ** 2) * d_em)
    myp = 44324.18 / (0.5 * (L ** 2) * d_em)
    Jzzp = 140067300 / (0.5 * (L ** 4) * d_em)
    Izzp = mp * (kzzp ** 2) + mp * (xGp ** 2)

    Mmat = np.zeros((3, 3))

    Mmat[0, 0] = mp + mxp
    Mmat[1, 1] = mp + myp
    Mmat[2, 2] = Izzp + Jzzp
    Mmat[1, 2] = mp * xGp
    Mmat[2, 1] = mp * xGp

    Mmatinv = np.linalg.inv(Mmat)

    tau = np.array([X, Y, N])

    vel_der = Mmatinv @ tau

    # Derivative of state vector
    vd = np.zeros(7)

    vd[0:3] = vel_der
    vd[3] = up * np.cos(psi) - vp * np.sin(psi)
    vd[4] = up * np.sin(psi) + vp * np.cos(psi)
    vd[5] = rp

    # # Commanded Rudder Angle
    # delta_c = KCS_rudder_angle(t, v)

    T_rud = 0.1  # Corresponds to a time constant of 0.1 * L / U_des = 2 seconds
    deltad = (delta_c - delta) / T_rud

    deltad_max = 5 * np.pi / 180 * (L / U_des)  # Maximum rudder rate of 5 degrees per second

    # Rudder rate saturation
    if np.abs(deltad) > deltad_max:
        deltad = np.sign(deltad) * deltad_max

    vd[6] = deltad

    return vd