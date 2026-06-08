import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np
import cantera as ct
import scipy.integrate as spi

#------------------------------------------------------------------------------------------------------------------------
#   Library of different ideal reactor models originally developed by Mike Bonheure (LCT). 
#------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Ideal continuously stirred tank reactor, gas-only
#-------------------------------------------------------------------------------

class CSTR:
    """
    Class to solve CSTR model, either transiently or to steady-state.

    Parameters
    ----------
    mechanism_file : string
        Name of Cantera input file (*.cti, *.yaml)
    gas_id : string
        Name of the gas phase in the mechanism file
    mdot : float
        Mass flow rate [kg/s]
    volume : float
        Volume of the reactor [m3]
    energy_type : string, optional
        Type of energy treatment: `adiabatic` (default), `isothermal`,
        `heat-transfer-surroundings`, `heat-flux`
    wall_area : float, optional
        Reactor wall area for heat transfer [m2]\n
        Used when energy_type = `heat-flux` or `heat-transfer-surroundings`
    heat_flux : cantera.Func1 object, optional
        Heat flux profile as a function of time [W/m2]\n
        Required when energy_type = `heat-flux`
    U : float, optional
        Global heat transfer coefficient [W/m2/K]\n
        Required when energy_type = `heat-transfer-surroundings`
    Tr : float, optional
        Temperature of surroundings [K]\n
        Required when energy_type = `heat-transfer-surroundings`
    kv : float, optional
        Valve constant
    valve_type : string, optional
        Type of valve for downstream reservoir.
        Use `P` for `PressureController`, otherwise `Valve` is used.



    Examples
    --------

    >>> myCSTR = CSTR(mechanism_file, gas_id, mass_flow_rate, volume)
    >>> myCSTR.set_inlet_TPY(T_in, P_in, Y_in)
    >>> results = myCSTR.transient_solve(endtime, 100)

    """

    def __init__(self, mechanism_file, gas_id, mdot, volume,
                    energy_type = 'adiabatic', wall_area = 0.0,
                    heat_flux = None, U = None, Tr = None, kv = 1e-5,
                    valve_type = 'P'):
        self.gas = ct.Solution(mechanism_file, gas_id)
        self.mdot = mdot
        self.V = volume
        self.A = wall_area
        self.energy_type = energy_type
        if self.energy_type == 'isothermal':
            self.isothermal = True
        elif self.energy_type == 'heat-flux':
            if heat_flux is None:
                raise Exception('Wall heat flux (heat_flux) not specified')
            self.hf = heat_flux
            self.env = ct.Reservoir(ct.Solution('air.xml'))
            self.env.thermo.TP = 298.15, ct.one_atm
        elif self.energy_type == 'heat-transfer-surroundings':
            if U is None:
                raise Exception('Heat transfer coefficient (U) not specified')
            elif Tr is None:
                raise Exception('Temperature surroundings (Tr) not specified')
            self.U = U
            self.Tr = Tr
            self.env = ct.Reservoir(ct.Solution('air.xml'))
            self.env.thermo.TP = self.Tr, ct.one_atm
        self.kv = kv
        self.valve_type = valve_type

    def initialize(self):
        self.upstream = ct.Reservoir(self.gas)
        self.cstr = ct.IdealGasReactor(self.gas, volume = self.V)
        if self.energy_type == 'isothermal':
            self.cstr.energy_enabled = False
        elif self.energy_type == 'heat-flux':
            self.wall = ct.Wall(self.cstr, self.env, Q = self.hf)
        elif self.energy_type == 'heat-transfer-surroundings':
            self.wall = ct.Wall(self.cstr, self.env, A = self.A, U = self.U)
        self.downstream = ct.Reservoir(self.gas)
        self.m = ct.MassFlowController(self.upstream,self. cstr,
                                       mdot = self.mdot)
        if self.valve_type == 'P':
            self.v = ct.PressureController(self.cstr, self.downstream,
                                           master = self.m, K = self.kv)
        else:
            self.v = ct.Valve(self.cstr, self.downstream, K = self.kv)

    def transient_solve(self, endtime, nsteps, steady = False,
                        solver_rtol = None, solver_atol = None):
        """
        Solve the CSTR model in time.

        NOTE: the initial state is the current state of the gas phase object,
        which has to be set prior to calling this `solve` function.

        Parameters
        ----------
        endtime : float
            The time over which to integrate the reactor model equations [s]
        nsteps : int
            Number of intervals in the output
        steady : bool, optional
            Optionally solves to steady state after initially solving transient
        solver_rtol : float, optional
            Relative tolerance for ODE solver
        solver_atol : float, optional
            Absolute tolerance for ODE solver

        Returns
        ------
        states : SolutionArray
            Object containing thermodynamic states as a function of time t

        """
        self.initialize()
        sim = ct.ReactorNet([self.cstr])
        if solver_rtol is not None:
            sim.rtol = solver_rtol
        if solver_atol is not None:
            sim.atol = solver_atol
        dt = endtime/nsteps
        states = ct.SolutionArray(self.gas, 1, extra= {'t': [0.0]})
        while sim.time < endtime:
            sim.advance(sim.time + dt)
            states.append(self.cstr.thermo.state, t = sim.time)

        if steady:
            sim.advance_to_steady_state()
            states.append(self.cstr.thermo.state, t = np.infty)

        return states

    def steady_solve(self, solver_rtol = None, solver_atol = None):
        """
        Solve the CSTR model in time.

        NOTE: the initial state is the current state of the gas phase object,
        which has to be set prior to calling this `solve` function.

        Parameters
        ----------
        endtime : float
            The time over which to integrate the reactor model equations [s]
        nsteps : int
            Number of intervals in the output
        solver_rtol : float, optional
            Relative tolerance for ODE solver
        solver_atol : float, optional
            Absolute tolerance for ODE solver

        Returns
        ------
        states : SolutionArray
            Object containing initial and steady-state thermodynamic states

        """
        self.initialize()
        sim = ct.ReactorNet([self.cstr])
        if solver_rtol is not None:
            sim.rtol = solver_rtol
        if solver_atol is not None:
            sim.atol = solver_atol
        states = ct.SolutionArray(self.gas, 1)
        sim.advance_to_steady_state()
        states.append(self.cstr.thermo.state)
        return states

    def set_inlet_TPX(self, T, P, X):
        self.gas.TPX = T, P, X
        self.T0 = self.gas.T
        self.p0 = self.gas.P
        self.y0 = self.gas.Y

    def set_inlet_TPY(self, T, P, Y):
        self.gas.TPY = T, P, Y
        self.T0 = self.gas.T
        self.p0 = self.gas.P
        self.y0 = self.gas.Y

#-------------------------------------------------------------------------------
#  Ideal continuously stirred tank reactor, pseudo-homogeneous catalytic
#-------------------------------------------------------------------------------

class surfaceCSTR:
    """
    Class to solve peudo-homogeneous CSTR model with catalytic chemistry,
    either transiently or to steady-state.

    Parameters
    ----------
    mechanism_file : string
        Name of Cantera input file (*.cti, *.yaml)
    gas_id : string
        Name of the gas phase in the mechanism file
    surf_id : string
        Name of the surface phase in the mechanism file
    mdot : float
        Mass flow rate [kg/s]
    volume : float
        Volume of the reactor [m3]
    epsB : float
        Bed voidage [m3_gas/m3_reactor]
    av : float
        Catalytic (internal) surface area per pellet volume [m2_cat/m3_pellet]
    epsC : float, optional
        Catalyst porosity [m3_pore/m3_pellet]
    energy_type : string, optional
        Type of energy treatment: `adiabatic` (default), `isothermal`,
        `heat-transfer-surroundings`, `heat-flux`
    wall_area : float, optional
        Reactor wall area for heat transfer [m2]\n
        Used when energy_type = `heat-flux` or `heat-transfer-surroundings`
    heat_flux : cantera.Func1 object, optional
        Heat flux profile as a function of time [W/m2]\n
        Required when energy_type = `heat-flux`
    U : float, optional
        Global heat transfer coefficient [W/m2/K]\n
        Required when energy_type = `heat-transfer-surroundings`
    Tr : float, optional
        Temperature of surroundings [K]\n
        Required when energy_type = `heat-transfer-surroundings`
    kv : float, optional
        Valve constant
    valve_type : string, optional
        Type of valve for downstream reservoir.
        Use `P` for `PressureController`, otherwise `Valve` is used.


    Examples
    --------

    >>> catCSTR = CSTR(mechanism_file, gas_id, surf_id, mass_flow_rate,
                       volume, bed_porosity, cat_area_per_volume)
    >>> catCSTR.set_inlet_TPX(T_in, P_in, Y_in)
    >>> results = catCSTR.transient_solve(endtime, 100)

    """

    def __init__(self, mechanism_file, gas_id, surf_id, mdot, volume, epsB, av,
                    epsC = 0.0, energy_type = 'adiabatic', wall_area = 0.0,
                    heat_flux = None, U = None, Tr = None, kv = 1e-5,
                    valve_type = 'P'):
        self.gas = ct.Solution(mechanism_file, gas_id)
        self.surf = ct.Interface(mechanism_file, surf_id, [self.gas])
        self.mdot = mdot
        self.epsB = epsB
        self.av = av
        self.epsC = epsC
        self.V = self.epsB*volume + (1.0 - self.epsB)*self.epsC*volume
        self.cat_area = (1.0 - self.epsB)*self.av*volume
        self.A = wall_area
        self.energy_type = energy_type
        if self.energy_type == 'isothermal':
            self.isothermal = True
        elif self.energy_type == 'heat-flux':
            if heat_flux is None:
                raise Exception('Wall heat flux (heat_flux) not specified')
            self.hf = heat_flux
            self.env = ct.Reservoir(ct.Solution('air.xml'))
            self.env.thermo.TP = 298.15, ct.one_atm
        elif self.energy_type == 'heat-transfer-surroundings':
            if U is None:
                raise Exception('Heat transfer coefficient (U) not specified')
            elif Tr is None:
                raise Exception('Temperature surroundings (Tr) not specified')
            self.U = U
            self.Tr = Tr
            self.env = ct.Reservoir(ct.Solution('air.xml'))
            self.env.thermo.TP = self.Tr, ct.one_atm
        self.kv = kv
        self.valve_type = valve_type

    def initialize(self):
        self.upstream = ct.Reservoir(self.gas)
        self.cstr = ct.IdealGasReactor(self.gas, volume = self.V)
        self.rsurf = ct.ReactorSurface(self.surf, self.cstr, A = self.cat_area)
        if self.energy_type == 'isothermal':
            self.cstr.energy_enabled = False
        elif self.energy_type == 'heat-flux':
            self.wall = ct.Wall(self.cstr, self.env, Q = self.hf)
        elif self.energy_type == 'heat-transfer-surroundings':
            self.wall = ct.Wall(self.cstr, self.env, A = self.A, U = self.U)
        self.downstream = ct.Reservoir(self.gas)
        self.m = ct.MassFlowController(self.upstream,self. cstr,
                                       mdot = self.mdot)
        if self.valve_type == 'P':
            self.v = ct.PressureController(self.cstr, self.downstream,
                                           master = self.m, K = self.kv)
        else:
            self.v = ct.Valve(self.cstr, self.downstream, K = self.kv)

    def transient_solve(self, endtime, nsteps, steady = False,
                        solver_rtol = None, solver_atol = None):
        """
        Solve the CSTR model in time.

        NOTE: the initial state is the current state of the gas phase object,
        which has to be set prior to calling this `solve` function.

        Parameters
        ----------
        endtime : float
            The time over which to integrate the reactor model equations [s]
        nsteps : int
            Number of intervals in the output
        steady : bool, optional
            Optionally solves to steady state after initially solving transient
        solver_rtol : float, optional
            Relative tolerance for ODE solver
        solver_atol : float, optional
            Absolute tolerance for ODE solver

        Returns
        ------
        states : SolutionArray
            Object containing thermodynamic states of the gas phase as a
            function of position z
        surfstates : SolutionArray
            Object containing thermodynamic states of the catalyst surface as
            a function of position z

        """
        self.initialize()
        sim = ct.ReactorNet([self.cstr])
        if solver_rtol is not None:
            sim.rtol = solver_rtol
        if solver_atol is not None:
            sim.atol = solver_atol
        dt = endtime/nsteps
        states = ct.SolutionArray(self.gas, 1, extra= {'t': [0.0]})
        surfstates = ct.SolutionArray(self.surf, 1, extra={'t': [0.0]})
        while sim.time < endtime:
            sim.advance(sim.time + dt)
            states.append(self.cstr.thermo.state, t = sim.time)
            surfstates.append(self.rsurf.thermo.state, t = sim.time)

        if steady:
            sim.advance_to_steady_state()
            states.append(self.cstr.thermo.state, t = np.infty)
            surfstates.append(self.rsurf.thermo.state, t = np.infty)

        return states, surfstates

    def steady_solve(self, solver_rtol = None, solver_atol = None):
        """
        Solve the CSTR model in time.

        NOTE: the initial state is the current state of the gas phase object,
        which has to be set prior to calling this `solve` function.

        Parameters
        ----------
        endtime : float
            The time over which to integrate the reactor model equations [s]
        nsteps : int
            Number of intervals in the output
        solver_rtol : float, optional
            Relative tolerance for ODE solver
        solver_atol : float, optional
            Absolute tolerance for ODE solver

        Returns
        ------
        states : SolutionArray
            Object containing initial and steady-state thermodynamic states
            of the gas phase
        surfstates : SolutionArray
            Object containing initial and steady-state thermodynamic states
            of the catalyst surface

        """
        self.initialize()
        sim = ct.ReactorNet([self.cstr])
        if solver_rtol is not None:
            sim.rtol = solver_rtol
        if solver_atol is not None:
            sim.atol = solver_atol
        states = ct.SolutionArray(self.gas, 1)
        surfstates = ct.SolutionArray(self.surf, 1)
        sim.advance_to_steady_state()
        states.append(self.cstr.thermo.state)
        surfstates.append(self.rsurf.thermo.state)
        return states

    def set_inlet_TPX(self, T, P, X):
        self.gas.TPX = T, P, X
        self.T0 = self.gas.T
        self.p0 = self.gas.P
        self.y0 = self.gas.Y

    def set_inlet_TPY(self, T, P, Y):
        self.gas.TPY = T, P, Y
        self.T0 = self.gas.T
        self.p0 = self.gas.P
        self.y0 = self.gas.Y

#-------------------------------------------------------------------------------
#  Ideal plug flow reactor, gas-only
#-------------------------------------------------------------------------------

class PFR:
    """
    Class to solve the plug flow reactor model equations for a gas-only
    reactor.

    Parameters
    ----------
    mechanism_file : string
        Name of Cantera input file (*.cti, *.yaml)
    gas_id : string
        Name of the gas phase in the mechanism file
    mdot : float
        Mass flow rate [kg/s]
    diam : float
        Diameter of the reactor [m]
    energy_type : string, optional
        Type of energy treatment: `adiabatic` (default), `isothermal`,
        `heat-transfer-surroundings`, `heat-flux-profile`
    heat_flux : cantera.Func1 object, optional
        Heat flux profile as a function of axial coordinate [W/m2]\n
        Required when energy_type = `heat-flux-profile`
    U : float, optional
        Global heat transfer coefficient [W/m2/K]\n
        Required when energy_type = `heat-transfer-surroundings`
    Tr : float, optional
        Temperature of surroundings [K]\n
        Required when energy_type = `heat-transfer-surroundings`
    friction_factor : callable, optional
        Function to calculate the Darcy friction factor
        `fun(Re) -> float`
        where `Re` is the Reynolds number\n
        If set to `None` (default), the pressure equation is not solved


    Examples
    --------

    >>> myPFR = PFR(mechanism_file, gas_id, mass_flow_rate, diameter)
    >>> myPFR.set_inlet_TPY(T_in, P_in, Y_in)
    >>> results = myPFR.solve(length, 100)

    """

    def __init__(self, mechanism_file, gas_id, mdot, diam,
                    energy_type = 'adiabatic', heat_flux = None,
                    U = None, Tr = None, friction_factor = None):
        self.gas = ct.Solution(mechanism_file, gas_id)
        self.mdot = mdot
        self.diam = diam
        self.A = np.pi*diam**2.0/4.0
        self.P = np.pi*diam
        self.energy_type = energy_type
        if self.energy_type == 'isothermal':
            self.isothermal = True
        elif self.energy_type == 'heat-flux-profile':
            if heat_flux is None:
                raise Exception('Wall heat flux (heat_flux) not specified')
            self.hf = heat_flux
        elif self.energy_type == 'heat-transfer-surroundings':
            if U is None:
                raise Exception('Heat transfer coefficient (U) not specified')
            elif Tr is None:
                raise Exception('Temperature surroundings (Tr) not specified')
            self.U = U
            self.Tr = Tr
        self.fD = friction_factor

    def ODE(self, z, y):
        """
        The ODE function

        Returns
        ------

        .. math::
            \\frac{\\partial \\vec{y}}{\\partial z} = f(z,\\vec{y})

        with

        .. math::
            \\vec{y} = \\{T, p, \\vec{Y_k}\\}

        .. math::
            c_{p,mix}\\dot{m} \\frac{\\partial T}{\\partial z}
                    = - A \\sum_{k=1}^{N_g} h_k(T) R_{g,k} W_k
                        \\left[+ P q_w\\right] \\left[+ P U (T_r - T)\\right]

        .. math::
            \\frac{\\partial p}{\\partial z} = - f_D\\frac{P}{8 A} \\rho U^2

        .. math::
            \\dot{m} \\frac{\\partial Y_k}{\\partial z} = A R_{g,k} W_k

        """

        self.set_state(y)
        wdot = self.gas.net_production_rates
        dYdz = wdot * self.gas.molecular_weights / self.mdot * self.A

        dTdz = 0.0
        if self.energy_type != 'isothermal':
            dTdz = - np.dot(self.gas.partial_molar_enthalpies, wdot) * self.A
            if self.energy_type == 'heat-flux-profile':
                dTdz += self.hf(z) * self.P
            elif self.energy_type == 'heat-transfer-surroundings':
                dTdz += self.U * self.P * (self.Tr - self.gas.T)
            dTdz /= self.mdot * self.gas.cp_mass

        dpdz = 0.0
        if self.fD is not None:
            rho = self.gas.density
            u = self.mdot / rho / self.A
            mu = self.gas.viscosity
            Re = rho * self.diam * u / mu
            dpdz = - self.fD(Re) * self.P / self.A * rho / 8.0 * u**2.0

        return np.hstack((dTdz, dpdz, dYdz))

    def set_state(self, y):
        """
        Set the state vector
            \\vec{y} = \\{T, p, \\vec{Y_k}\\}
        """
        self.gas.set_unnormalized_mass_fractions(y[2:])
        self.gas.TP = y[0], y[1]

    def set_inlet_TPX(self, T, P, X):
        self.gas.TPX = T, P, X

    def set_inlet_TPY(self, T, P, Y):
        self.gas.TPY = T, P, Y

    def solve(self, length, nsteps, stop_rt = np.infty,
                    solver_rtol = 1e-9, solver_atol = 1e-15):
        """
        Solve the reactor model.

        NOTE: the initial state is the current state of the gas phase object,
        which has to be set prior to calling this `solve` function.

        Parameters
        ----------
        length : float
            The length of the reactor to integrate the equations over [m]
        nsteps : int
            Number of intervals in the output
        stop_rt : float, optional
            Optionally, integrate the reactor models until reaching this residence time.
            The solver will stop if either the residence time or specified
            reactor length is reached.
        solver_rtol : float, optional
            Relative tolerance for ODE solver
        solver_atol : float, optional
            Absolute tolerance for ODE solver

        Returns
        ------
        states : SolutionArray
            Object containing thermodynamic states as a function of position z

        """
        y0 = np.hstack((self.gas.T, self.gas.P, self.gas.Y))
        restime = 0.0
        u = self.mdot / self.gas.density / self.A
        dz = length / nsteps if stop_rt > 1e6 else u * stop_rt / nsteps
        states = ct.SolutionArray(self.gas, 1, extra=
                                                {'z': [0.0], 'tau':[restime],
                                                'velocity': [u]})
        solver = spi.ode(self.ODE)
        solver.set_integrator('vode', method = 'bdf', with_jacobian = True,
                                nsteps = 1000, order = 5,
                                atol = solver_atol, rtol = solver_rtol)
        solver.set_initial_value(y0, 0.0)
        while solver.successful() and solver.t < length and restime < stop_rt:
            solver.integrate(solver.t + dz)
            self.set_state(solver.y)
            u = self.mdot / self.gas.density / self.A
            restime = restime + dz/u
            states.append(self.gas.state, z = solver.t, tau = restime, velocity = u)

        if not(solver.successful()):
            print('Simulation was unsuccessful')

        return states

#-------------------------------------------------------------------------------
#  Ideal plug flow reactor, pseudo-homogeneous catalytic
#-------------------------------------------------------------------------------

class surfacePFR:
    """
    Class to solve the pseudo-homogeneous plug flow reactor model equations
    for a catalytic reactor.


    Parameters
    ----------
    mechanism_file : string
        Name of Cantera input file (*.cti, *.yaml)
    gas_id : string
        Name of the gas phase in the mechanism file
    surf_id : string
        Name of the surface phase in the mechanism file
    mdot : float
        Mass flow rate [kg/s]
    diam : float
        Diameter of the reactor [m]
    epsB : float
        Bed voidage [m3_gas/m3_reactor]
    av : float
        Catalytic (internal) surface area per pellet volume [m2_cat/m3_pellet]
    epsC : float, optional
        Catalyst porosity [m3_pore/m3_pellet]
    energy_type : string, optional
        Type of energy treatment: `adiabatic` (default), `isothermal`,
        `heat-transfer-surroundings`, `heat-flux-profile`
    heat_flux : cantera.Func1 object, optional
        Heat flux profile as a function of axial coordinate [W/m2]\n
        Required when energy_type = `heat-flux-profile`
    U : float, optional
        Global heat transfer coefficient [W/m2/K]\n
        Required when energy_type = `heat-transfer-surroundings`
    Tr : float, optional
        Temperature of surroundings [K]\n
        Required when energy_type = `heat-transfer-surroundings`
    friction_factor :  tuple, optional
        Darcy-Forccheimer coefficients, needed to solve the pressure equation

        .. math::
            \\frac{\\partial p}{\\partial z} =
                        - \\left( \\mu D + \\frac{1}{2}\\rho F U \\right) U

        If set to `None` (default), the pressure equation is not solved


    Examples
    --------

    >>> catPFR = surfacePFR(mechanism_file, gas_id, surf_id, mass_flow_rate,
                             diam, epsB, area_per_volume, U = 300.0, Tr = 298.0,
                             energy_type = 'heat-transfer-surroundings')
    >>> catPFR.set_inlet_TPX(T_in, P_in, X_in)
    >>> states, surfstates = catPFR.solve(length, 100)

    """

    def __init__(self, mechanism_file, gas_id, surf_id, mdot, diam, epsB, av,
                    epsC = 0.0, energy_type = 'adiabatic', heat_flux = None,
                    U = None, Tr = None, friction_factors = None):
        self.gas = ct.Solution(mechanism_file, gas_id)
        self.surf = ct.Interface(mechanism_file, surf_id, [self.gas])
        self.mdot = mdot
        self.diam = diam
        self.epsB = epsB
        self.av = av
        self.epsC = epsC
        self.A = np.pi*diam**2.0/4.0
        self.P = np.pi*diam
        self.energy_type = energy_type
        if self.energy_type == 'isothermal':
            self.isothermal = True
        elif self.energy_type == 'heat-flux-profile':
            if heat_flux is None:
                raise Exception('Wall heat flux (heat_flux) not specified')
            self.hf = heat_flux
        elif self.energy_type == 'heat-transfer-surroundings':
            if U is None:
                raise Exception('Heat transfer coefficient (U) not specified')
            elif Tr is None:
                raise Exception('Temperature surroundings (Tr) not specified')
            self.U = U
            self.Tr = Tr
        self.DF = friction_factors
        if self.DF is not None:
            self.D = self.DF[0]
            self.F = self.DF[1]

    def ODE(self, z, y):
        """
        The ODE function

        Returns
        ------

        .. math::
            \\frac{\\partial \\vec{y}}{\\partial z} = f(z,\\vec{y})

        with

        .. math::
            \\vec{y} = \\{T, p, \\vec{Y_k}\\}

        .. math::
            c_{p,mix}\\dot{m} \\frac{\\partial T}{\\partial z}
                    = - A \\sum_{k=1}^{N_g} h_k(T) R_{k} W_k
                        \\left[+ P q_w\\right] \\left[+ P U (T_r - T)\\right]

        .. math::
            \\frac{\\partial p}{\\partial z} =
                        - \\left( \\mu D + \\frac{1}{2}\\rho F U \\right) U

        .. math::
            \\dot{m} \\frac{\\partial Y_k}{\\partial z} = A R_{k} W_k

        .. math::
            R_{k} = \\varepsilon_B R_{g,k} + (1.0 - \\varepsilon_B )
                                        (\\varepsilon_C R_{g,k} + a_V R_{s,k})

        """

        self.set_state(y)
        wdot = self.gas.net_production_rates
        self.surf.advance_coverages_to_steady_state()
        sdot = self.surf.get_net_production_rates(self.gas)
        rdot = self.epsB * wdot + (1.0 - self.epsB) * (self.epsC * wdot + self.av * sdot)
        dYdz = rdot * self.gas.molecular_weights / self.mdot * self.A

        dTdz = 0.0
        if self.energy_type != 'isothermal':
            dTdz = - np.dot(self.gas.partial_molar_enthalpies, rdot) * self.A
            if self.energy_type == 'heat-flux-profile':
                dTdz += self.hf(z) * self.P
            elif self.energy_type == 'heat-transfer-surroundings':
                dTdz += self.U * self.P * (self.Tr - self.gas.T)
            dTdz /= self.mdot * self.gas.cp_mass

        dpdz = 0.0
        if self.DF is not None:
            rho = self.gas.density
            u = self.mdot / rho / self.A
            mu = self.gas.viscosity
            dpdz = - (self.D * mu + 0.5 * self.F * rho * u) * u

        return np.hstack((dTdz, dpdz, dYdz))

    def set_state(self, y):
        """
        Set the state vector
            \\vec{y} = \\{T, p, \\vec{Y_k}\\}
        """
        self.gas.set_unnormalized_mass_fractions(y[2:])
        self.gas.TP = y[0], y[1]
        self.surf.TP = y[0], y[1]
        self.surf.advance_coverages_to_steady_state()

    def set_inlet_TPX(self, T, P, X):
        self.gas.TPX = T, P, X

    def set_inlet_TPY(self, T, P, Y):
        self.gas.TPY = T, P, Y

    def solve(self, length, nsteps, stop_rt = np.infty,
                    solver_rtol = 1e-9, solver_atol = 1e-15):
        """
        Solve the reactor model.

        NOTE: the initial state is the current state of the gas phase object,
        which has to be set prior to calling this `solve` function.

        Parameters
        ----------
        length : float
            The length of the reactor to integrate the equations over [m]
        nsteps : int
            Number of intervals in the output
        stop_rt : float, optional
            Optionally, integrate the reactor models until reaching this residence time.
            The solver will stop if either the residence time or specified
            reactor length is reached.
        solver_rtol : float, optional
            Relative tolerance for ODE solver
        solver_atol : float, optional
            Absolute tolerance for ODE solver

        Returns
        ------
        states : SolutionArray
            Object containing thermodynamic states of the gas phase as a
            function of position z
        surfstates : SolutionArray
            Object containing thermodynamic states of the catalyst surface as
            a function of position z

        """
        y0 = np.hstack((self.gas.T, self.gas.P, self.gas.Y))
        restime = 0.0
        u = self.mdot / self.gas.density / self.A / self.epsB
        dz = length / nsteps if stop_rt > 1e6 else u * stop_rt / nsteps
        states = ct.SolutionArray(self.gas, 1, extra=
                                                {'z': [0.0], 'tau':[restime],
                                                'velocity': [u]})
        surfstates = ct.SolutionArray(self.surf, 1, extra={'z': [0.0]})
        solver = spi.ode(self.ODE)
        solver.set_integrator('vode', method = 'bdf', with_jacobian = True,
                                nsteps = 1000, order = 5,
                                atol = solver_atol, rtol = solver_rtol)
        solver.set_initial_value(y0, 0.0)
        while solver.successful() and solver.t < length and restime < stop_rt:
            solver.integrate(solver.t + dz)
            self.set_state(solver.y)
            u = self.mdot / self.gas.density / self.A / self.epsB
            restime = restime + dz/u
            states.append(self.gas.state, z = solver.t, tau = restime, velocity = u)
            surfstates.append(self.surf.state, z=solver.t)

        if not(solver.successful()):
            print('Simulation was unsuccessful')

        return states, surfstates

#-------------------------------------------------------------------------------
#  Ideal plug flow reactor, custom reaction rates
#-------------------------------------------------------------------------------

class customPFR:
    """
    Class to solve the plug flow reactor model equations for a gas-only
    reactor. This implementation relies on a user-defined function to calculate
    the reaction rates, given the thermodynamic state of the gas phase.

    Parameters
    ----------
    mechanism_file : string
        Name of Cantera input file (*.cti, *.yaml)
    gas_id : string
        Name of the gas phase in the mechanism file
    mdot : float
        Mass flow rate [kg/s]
    diam : float
        Diameter of the reactor [m]
    reaction_rates : callable
        Function `fun(phase, epsB, av, epsC) -> ndarray` to calculate the net
        production rates, given the thermodynamic state of the gas `phase`, the
        bed porosity, the area-per-volume and catalyst porosity.\n
        Returns an array with the net production rate of all species.
    energy_type : string, optional
        Type of energy treatment: `adiabatic` (default), `isothermal`,
        `heat-transfer-surroundings`, `heat-flux-profile`
    heat_flux : cantera.Func1 object, optional
        Heat flux profile as a function of axial coordinate [W/m2]\n
        Required when energy_type = `heat-flux-profile`
    U : float, optional
        Global heat transfer coefficient [W/m2/K]\n
        Required when energy_type = `heat-transfer-surroundings`
    Tr : float, optional
        Temperature of surroundings [K]\n
        Required when energy_type = `heat-transfer-surroundings`
    friction_factor :  tuple, optional
        Darcy-Forccheimer coefficients, needed to solve the pressure equation

        .. math::
            \\frac{\\partial p}{\\partial z} =
                        - \\left( \\mu D + \\frac{1}{2}\\rho F U \\right) U

        If set to `None` (default), the pressure equation is not solved


    Examples
    --------

    >>> def rates(phase):
            rr = np.zeros(phase.n_species)
            rr[phase.species_index('CH4')] =
            return rr
    >>> myPFR = customPFR(mechanism_file, gas_id, mass_flow_rate, diam, rates)
    >>> myPFR.set_inlet_TPY(T_in, P_in, Y_in)
    >>> results = myPFR.solve(length, 100)

    """

    def __init__(self, mechanism_file, gas_id, mdot, diam, reaction_rates,
                    energy_type = 'adiabatic',
                    heat_flux = None, U = None, Tr = None,
                    friction_factors = None):
        self.gas = ct.Solution(mechanism_file, gas_id)
        self.mdot = mdot
        self.diam = diam
        self.rr = reaction_rates
        self.A = np.pi*diam**2.0/4.0
        self.P = np.pi*diam
        self.energy_type = energy_type
        if self.energy_type == 'isothermal':
            self.isothermal = True
        elif self.energy_type == 'heat-flux-profile':
            if heat_flux is None:
                raise Exception('Wall heat flux (heat_flux) not specified')
            self.hf = heat_flux
        elif self.energy_type == 'heat-transfer-surroundings':
            if U is None:
                raise Exception('Heat transfer coefficient (U) not specified')
            elif Tr is None:
                raise Exception('Temperature surroundings (Tr) not specified')
            self.U = U
            self.Tr = Tr
        self.DF = friction_factors
        if self.DF is not None:
            self.D = self.DF[0]
            self.F = self.DF[1]

    def ODE(self, z, y):
        """
        The ODE function

        Returns
        ------

        .. math::
            \\frac{\\partial \\vec{y}}{\\partial z} = f(z,\\vec{y})

        with

        .. math::
            \\vec{y} = \\{T, p, \\vec{Y_k}\\}

        .. math::
            c_{p,mix}\\dot{m} \\frac{\\partial T}{\\partial z}
                    = - A \\sum_{k=1}^{N_g} h_k(T) R_{k} W_k
                        \\left[+ P q_w\\right] \\left[+ P U (T_r - T)\\right]

        .. math::
            \\frac{\\partial p}{\\partial z} =
                        - \\left( \\mu D + \\frac{1}{2}\\rho F U \\right) U

        .. math::
            \\dot{m} \\frac{\\partial Y_k}{\\partial z} = A R_{k} W_k

        """

        self.set_state(y)
        rdot = self.rr(self.gas)
        dYdz = rdot * self.gas.molecular_weights / self.mdot * self.A

        dTdz = 0.0
        if self.energy_type != 'isothermal':
            dTdz = - np.dot(self.gas.partial_molar_enthalpies, rdot) * self.A
            if self.energy_type == 'heat-flux-profile':
                dTdz += self.hf(z) * self.P
            elif self.energy_type == 'heat-transfer-surroundings':
                dTdz += self.U * self.P * (self.Tr - self.gas.T)
            dTdz /= self.mdot * self.gas.cp_mass

        dpdz = 0.0
        if self.DF is not None:
            rho = self.gas.density
            u = self.mdot / rho / self.A
            mu = self.gas.viscosity
            dpdz = - (self.D * mu + 0.5 * self.F * rho * u) * u

        return np.hstack((dTdz, dpdz, dYdz))

    def set_state(self, y):
        """
        Set the state vector
            \\vec{y} = \\{T, p, \\vec{Y_k}\\}
        """
        self.gas.set_unnormalized_mass_fractions(y[2:])
        self.gas.TP = y[0], y[1]

    def set_inlet_TPX(self, T, P, X):
        self.gas.TPX = T, P, X

    def set_inlet_TPY(self, T, P, Y):
        self.gas.TPY = T, P, Y

    def solve(self, length, nsteps, stop_rt = np.infty,
                    solver_rtol = 1e-9, solver_atol = 1e-15):
        """
        Solve the reactor model.

        NOTE: the initial state is the current state of the gas phase object,
        which has to be set prior to calling this `solve` function.

        Parameters
        ----------
        length : float
            The length of the reactor to integrate the equations over [m]
        nsteps : int
            Number of intervals in the output
        stop_rt : float, optional
            Optionally, integrate the reactor models until reaching this residence time.
            The solver will stop if either the residence time or specified
            reactor length is reached.
        solver_rtol : float, optional
            Relative tolerance for ODE solver
        solver_atol : float, optional
            Absolute tolerance for ODE solver

        Returns
        ------
        states : SolutionArray
            Object containing thermodynamic states as a function of position z

        """
        y0 = np.hstack((self.gas.T, self.gas.P, self.gas.Y))
        restime = 0.0
        u = self.mdot / self.gas.density 
        dz = length / nsteps if stop_rt > 1e6 else u * stop_rt / nsteps
        r = self.rr(self.gas)   #self.rr(self.gas) is CRACKSIM_rates_DLL(self.gas) mol/l/s
        rr = r*self.gas.molecular_weights
        rates = []
        h = []
        rates.append(rr.tolist())
        h_0 = self.gas.standard_enthalpies_RT * self.gas.T * ct.gas_constant  # J/kmol
        h.append(h_0.tolist())
        S = -np.dot(h_0,r)
        massflow = self.mdot
        cp = self.gas.cp_mass
        W = self.gas.mean_molecular_weight
        states = ct.SolutionArray(self.gas, 1, extra=
                                                {'z': [0.0], 'tau':[restime],
                                                'velocity': [u], 'energy_source_term': [S], 'mass_flow':[massflow], 'Cp_mass':[cp], 'Molecular_weight':[W]}) 
        solver = spi.ode(self.ODE)
        solver.set_integrator('vode', method = 'bdf', with_jacobian = True,
                                nsteps = 1000, order = 5,
                                atol = solver_atol, rtol = solver_rtol)
        solver.set_initial_value(y0, 0.0)
        while solver.successful() and solver.t < length and restime < stop_rt:
            solver.integrate(solver.t + dz)
            self.set_state(solver.y)
            u = self.mdot / self.gas.density
            restime = restime + dz/u
            r = self.rr(self.gas)
            rr = r*self.gas.molecular_weights
            rates.append(rr.tolist())
            h_0 = self.gas.standard_enthalpies_RT * self.gas.T * ct.gas_constant
            h.append(h_0.tolist())
            S = -np.dot(h_0,r)
            massflow = self.mdot
            cp = self.gas.cp_mass
            W = self.gas.mean_molecular_weight
            states.append(self.gas.state, z = solver.t, tau = restime, velocity = u, energy_source_term = S, mass_flow = massflow, Cp_mass = cp, Molecular_weight = W) 
            #print(solver.t)

        if not(solver.successful()):
            print('Simulation was unsuccessful')

        return states,rates,h

#------------------------------------------------------------------------------------------------------------------------
#   electrified Rotor-Stator Reactor/ Rotodynamic Reactor (eRSR/RDR) - BASE MODEL
#   Model originally developed by Mike Bonheure (LCT). For more info see thesis Mike and/or Thomas De Witte. 
#------------------------------------------------------------------------------------------------------------------------

class ESC_BM:
    """
    Class to solve electric steam cracking reactors which are based on turbomachinery 
    (see for  example,  https://coolbrook.com/). This implementation relies on a 
    user-defined function to calculate the reaction rates, given the thermodynamic state of the gas phase.

    Modeling assumptions
    ----------
    1) Vaneless space is assumed to be an ideal plug flow reactor. Depending on the type of reactor, this assumption is not 
       entirely true (i.e., occurence of lateral mixing between stream tubes of different age). 
    2) The reactor operates ISOBARIC (i.e., the static pressure is assumed to be constant throughout the reactor). 
       This is justified based on the fact that these type of machines are design as such that the pressure increase 
       (which is traditionally seen in a compressor) is nullified by introducing losses (i.e. entropy increase). Hence, accross
       the entire reactor (i.e. inlet - outlet) the pressure will hardly change. 
       
       Of course it must be acknowledged that locally static pressure changes occur but these are not accounted for in the 
       base-model. 
       
    3) The reactor operates ADIABATIC (i.e., there is no heat transfer to the environment).
    4) The code uses ODE's in function of residence time instead of axial position (i.e., like traditional PFR PDE's). 
       This because this approach allows to develop an general applicable code for all turbomachinery-based electric 
       steam crackers in which axial location not always makes physically sense (e.g. toroidal design of coolbrook). 
       
       NOTE: this does not make the equations transient in nature. The residence-time is merely a variable just like axial 
       location. A truly transient PFR, would require PDE's insteady of ODE's, which is not the case here as for 
       the derivations of the modeling equations it is relied upon the steady-state reynolds transport theorem.
    5) The power input per stage is modeled as a simple temperature increase, reduced by the occuring reactions.
       This value can be chosen ad-hoc or based on CFD or experimental results. 
    6) The coil outlet temperature is used as parameter to track the cracking severity. This method allows to determine
       how many stages must be implemented to reach the targeted severity. 
    7) The base model assumes that the cross sectional area does not change, as a consequence you must consider -for the
       physical interpretation- that an indentical residence time in each vaneless space implies that the length of the 
       vaneless space increases as a consequence of the cracking reactions and the reactor operation.

    Parameters
    ----------
    mechanism_file : string
        Name of Cantera input file (*.cti, *.yaml)
    gas_id : string
        Name of the gas phase in the mechanism file
    COT : float
        Coil Outlet Temperature [K] as measure for cracking severity\n
    RT_stage: float
        Residence time per stage [ms]\n
        Stage is defined here as both the vaned as vaneless section
    Vanes_Multiplier: float
        Average relative time which the fluid resides within the vaned section [/]\n
        Expressed in relative terms of the total RT_Stage\n
        (e.g. Vanes_Multiplier = 0.1, implies 10% of the total RT_stage is within the vaned section)
    RT_Multiplier: float
        Value with which the residence time in the vaneless section will be multiplied in the last section [/]\n
        This allows to account for the transport from the reactor to the TLE.   
    power_input : string, optional   
        Type of power addition to mimic rotor: 'Temperature'(default), 'Power'\n
    power: float, optional
        (total) enthalpy increase (see Euler's Turbomachine Equation)\n
        Required when power_input = `Power`
    T_rise_stage : float, optional 
        Static temperature increase in vaned section \n
        Required when power_input = `Temperature`
    reaction_vanes :   string, optional 
        Activating the occurence of reactions in the vaned section: 'True' (default), 'False'
    energy_type : string, optional
        Type of energy treatment: `adiabatic` (default),`heat-transfer-surroundings`
    U : float, optional
        Global heat transfer coefficient [W/m2/K]\n
        Required when energy_type = `heat-transfer-surroundings`
    Tr : float, optional
        Temperature of surroundings [K]\n
        Required when energy_type = `heat-transfer-surroundings`
    diam : float, optional
        Shroud diameter of the reactor [m]
        Required when energy_type = `heat-transfer-surroundings`

    Examples
    --------
    >>> # Operating conditions
    >>>     T_in = 600.0 # temperature [K]
    >>>     P_in = 2.0e5 # pressure [Pa]
    >>>     Y_in = {'C2H6':1.0, 'H2O':0.326}  # mass fractions
    
    >>> # Parameter setting
    >>>     COT = 1000
    >>>     RT_stage = 2
    >>>     Vanes_Multiplier = 0.1 
    >>>     RT_Multiplier = 1.5
    >>>     T_rise_stage = 60

    >>> # Create PFR object and solve - without pressure drop
    >>>     propane_ersr_noDP = customESC_BM(reaction_mechanism, gas_id, COT, RT_stage,Vanes_Multiplier, RT_Multiplier,
                                     power_input  = 'Temperature', T_rise_stage=T_rise_stage)
    >>>     propane_ersr_noDP.set_inlet_TPY(T_in, P_in, Y_in)
    >>>     states_noDP_ersr = propane_ersr_noDP.solve(propane_ersr_noDP)
    >>>     states_noDP_ersr=states_noDP_ersr()

    """

    def __init__(self, mechanism_file, gas_id, COT, RT_stage, Vanes_Multiplier, RT_Multiplier,
                 power_input = 'Temperature', power = None, T_rise_stage = None, reaction_vanes = True, 
                 energy_type = 'adiabatic', diam = None, heat_flux = None, U = None, Tr = None):
        
        
        self.gas = ct.Solution(mechanism_file, gas_id)
        self.reaction_vanes = reaction_vanes
        self.RT_stage = RT_stage
        self.Vanes_Multiplier = Vanes_Multiplier
        self.RT_Multiplier = RT_Multiplier
        self.COT = COT
        
        self.energy_type = energy_type
        if self.energy_type == 'heat-transfer-surroundings':
            if U is None:
                raise Exception('Heat transfer coefficient (U) not specified')
            elif diam is None: 
                raise Exception('Diameter (diam) not specified')
            elif Tr is None:
                raise Exception('Temperature surroundings (Tr) not specified')
            self.U = U
            self.Tr = Tr
            self.diam = diam
            self.P = np.pi*diam
            
        self.power_input = power_input
        if self.power_input == 'Temperature': 
            if T_rise_stage is None:
                raise Exception('Temperature increase due to vaned section (T_rise_stage) not specified')
            self.T_rise_stage = T_rise_stage
        elif self.power_input == 'Power': 
            if power is None:
                raise Exception('Power from rotor to fluid (power) not specified')
            self.power = power
        else: 
            raise Exception('The specified property of power_input is incorrect! Please correct the input.')

    def set_state(self, y):
        """
        Set the state vector
            \\vec{y} = \\{T, p, \\vec{Y_k}\\}
        """

        self.gas.set_unnormalized_mass_fractions(y[2:])
        self.gas.TP = y[0], y[1]

    def set_inlet_TPX(self, T, P, X):
        self.gas.TPX = T, P, X

    def set_inlet_TPY(self, T, P, Y):
        self.gas.TPY = T, P, Y                

    
    # work with nested classes to make sure that the correct solver is used each time one is using this
    class ODE_Vanes: 
        def __init__(self, outer_instance, tau_0, tau):
            self.outer_instance = outer_instance
            
            self.gas = self.outer_instance.gas
            self.reaction_vanes = self.outer_instance.reaction_vanes
            self.RT_stage = self.outer_instance.RT_stage
            self.Vanes_Multiplier = self.outer_instance.Vanes_Multiplier
            self.RT_Multiplier = self.outer_instance.RT_Multiplier
            self.COT = self.outer_instance.COT
            self.energy_type = self.outer_instance.energy_type
            if self.energy_type == 'heat-transfer-surroundings':
                self.U = self.outer_instance.U
                self.Tr = self.outer_instance.Tr
                self.diam = self.outer_instance.diam
                self.P = self.outer_instance.P
            
            self.power_input = self.outer_instance.power_input
            if self.power_input == 'Temperature': 
                self.T_rise_stage = self.outer_instance.T_rise_stage
            if self.power_input == 'Power': 
                self.power = self.outer_instance.power
            
            self.tau_ini = tau_0
            self.tau = tau
            
        def set_state(self, y):
            """
            Set the state vector
                \\vec{y} = \\{T, p, \\vec{Y_k}\\}
            """

            self.gas.set_unnormalized_mass_fractions(y[2:])
            self.gas.TP = y[0], y[1]

        def set_inlet_TPX(self, T, P, X):
            self.gas.TPX = T, P, X

        def set_inlet_TPY(self, T, P, Y):
            self.gas.TPY = T, P, Y    
        
        def __call__(self, z, y):  
            """
            The ODE functions for the VANED section of the reactor. 

            Returns
            ------

            .. math::
                \\frac{\\partial \\vec{y}}{\\partial z} = f(z,\\vec{y})

            with

            .. math::
                \\vec{y} = \\{T, p, \\vec{Y_k}\\}

            .. math::
                c_{p,mix}\\dot{m} \\frac{\\partial T}{\\partial z}
                        = - A \\sum_{k=1}^{N_g} h_k(T) R_{k} W_k
                            \\left[+ P q_w\\right] \\left[+ P U (T_r - T)\\right]

            .. math::
                \\frac{\\partial p}{\\partial z} =
                            - \\left( \\mu D + \\frac{1}{2}\\rho F U \\right) U

            .. math::
                \\dot{m} \\frac{\\partial Y_k}{\\partial z} = A R_{k} W_k

            """

            # NOTE: z is the residence time            
            
            self.set_state(y)
            wdot = self.gas.net_production_rates
            #Conservation of species
            dYdz = [0]*len(wdot)
            if  self.reaction_vanes == True:
                dYdz = wdot * self.gas.molecular_weights / self.gas.density
            

            #Conservation of Energy
            dTdz = 0.0
            if self.energy_type != 'isothermal':
                if  self.reaction_vanes == True:
                    dTdz = (-np.dot(self.gas.partial_molar_enthalpies, wdot))
                if self.energy_type == 'heat-transfer-surroundings':
                    dTdz += self.U * self.P * (self.Tr - self.gas.T)
                dTdz /= (self.gas.cp_mass*self.gas.density_mass)

            # Additional term on the RHS must be included in the energy equation. This because the design of these reactors, 
            # relies on the transfer of kinetic energy into internal energy (i.e. temperature) through processes like
            # diffusion and loss introduction. 
            
                if self.power_input == 'Temperature': 
                    dTdz += self.T_rise_stage/(self.tau-self.tau_ini)

            # The implementation of Power term is based on the Eulers Turbomachine equation (i.e. Energy conservation) 
            # However, as Euler is an algebraic equation the power input term must be defined as such that it quantifies 
            # a change in states. This can be done by defining the power input as Power/massflow or the total enthalpy difference
            # accross a single stage. We than rely  on the local linear approximation to of the Euler equation to calculate the 
            # slope of the power in function of the residence time. his slope is than divided by the cp value (assuming ideal gas) 
            # to get this in terms of T.   
                elif self.power_input == 'Power': 
                    dTdz +=(1/self.gas.cp_mass)*(self.power/(self.tau-self.tau_ini))                                        
                
            #Conservation of momentum
            dpdz = 0.0

            #Create a single array containing the solutions of all the solved ODE's  
            return np.hstack((dTdz, dpdz, dYdz))
    
    class ODE_Vaneless: 
        def __init__(self, outer_instance):
            self.outer_instance = outer_instance
            
            self.gas = self.outer_instance.gas
            self.reaction_vanes = self.outer_instance.reaction_vanes
            self.RT_stage = self.outer_instance.RT_stage
            self.Vanes_Multiplier = self.outer_instance.Vanes_Multiplier
            self.RT_Multiplier = self.outer_instance.RT_Multiplier
            self.COT = self.outer_instance.COT
            self.energy_type = self.outer_instance.energy_type
            if self.energy_type == 'heat-transfer-surroundings':
                self.U = self.outer_instance.U
                self.Tr = self.outer_instance.Tr
                self.diam = self.outer_instance.diam
                self.P = self.outer_instance.P
            
            self.power_input = self.outer_instance.power_input
            if self.power_input == 'Temperature': 
                self.T_rise_stage = self.outer_instance.T_rise_stage
            if self.power_input == 'Power': 
                self.power = self.outer_instance.power

        def set_state(self, y):
            """
            Set the state vector
                \\vec{y} = \\{T, p, \\vec{Y_k}\\}
            """

            self.gas.set_unnormalized_mass_fractions(y[2:])
            self.gas.TP = y[0], y[1]

        def set_inlet_TPX(self, T, P, X):
            self.gas.TPX = T, P, X

        def set_inlet_TPY(self, T, P, Y):
            self.gas.TPY = T, P, Y    
                        
            
            
        def __call__(self, z, y):  
            
            """
            The ODE functions for the vaned section of the reactor. 

            Returns
            ------

            .. math::
                \\frac{\\partial \\vec{y}}{\\partial z} = f(z,\\vec{y})

            with

            .. math::
                \\vec{y} = \\{T, p, \\vec{Y_k}\\}

            .. math::
                c_{p,mix}\\dot{m} \\frac{\\partial T}{\\partial z}
                        = - A \\sum_{k=1}^{N_g} h_k(T) R_{k} W_k
                            \\left[+ P q_w\\right] \\left[+ P U (T_r - T)\\right]

            .. math::
                \\frac{\\partial p}{\\partial z} =
                            - \\left( \\mu D + \\frac{1}{2}\\rho F U \\right) U

            .. math::
                \\dot{m} \\frac{\\partial Y_k}{\\partial z} = A R_{k} W_k

            """
            # z is the residence time
            
            self.set_state(y)
            wdot = self.gas.net_production_rates

            #Conservation of species
            dYdz = wdot * self.gas.molecular_weights / self.gas.density

            #Conservation of Energy
            dTdz = 0.0
            if self.energy_type != 'isothermal':
                dTdz = (-np.dot(self.gas.partial_molar_enthalpies, wdot))
                if self.energy_type == 'heat-transfer-surroundings':
                    dTdz += self.U * self.P * (self.Tr - self.gas.T)
                dTdz /= (self.gas.cp_mass*self.gas.density_mass)

            #Conservation of momentum
            dpdz = 0.0

            #Create a single array containing the solutions of all the solved ODE's                                
            return np.hstack((dTdz, dpdz, dYdz))


    class solve:
        def __init__(self, outer_instance):
            self.outer_instance = outer_instance          
            
            self.gas = self.outer_instance.gas
            self.reaction_vanes = self.outer_instance.reaction_vanes
            self.RT_stage = self.outer_instance.RT_stage
            self.Vanes_Multiplier = self.outer_instance.Vanes_Multiplier
            self.RT_Multiplier = self.outer_instance.RT_Multiplier
            self.COT = self.outer_instance.COT
            self.energy_type = self.outer_instance.energy_type
            if self.energy_type == 'heat-transfer-surroundings':
                self.U = self.outer_instance.U
                self.Tr = self.outer_instance.Tr
                self.diam = self.outer_instance.diam
                self.P = self.outer_instance.P
            
            self.power_input = self.outer_instance.power_input
            if self.power_input == 'Temperature': 
                self.T_rise_stage = self.outer_instance.T_rise_stage
            if self.power_input == 'Power': 
                self.power = self.outer_instance.power
        

        def __call__(self,solver_rtol = 1e-9, solver_atol = 1e-15): 

            """
            Solve the reactor model.

            NOTE: the initial state is the current state of the gas phase object,
            which has to be set prior to calling this `solve` function.

            Parameters
            ----------
            solver_rtol : float, optional
                Relative tolerance for ODE solver
            solver_atol : float, optional
                Absolute tolerance for ODE solver

            Returns
            ------
            states : SolutionArray
                Object containing thermodynamic states as a function of position z

            """

            #variable initialization 
            tau_0 = 0.0
            number = 0
            max_stages = 1000 # This value can be changed! 
                              # but of course question yourself if a reactor with +1000 stages is feasible?                                 
            states = ct.SolutionArray(self.gas, 1, extra={'tau': [tau_0]})

            #list declarations used throughout the solving routine
            T_outlet = [self.gas.T]
            RT_Stage_Vanes =[]
            RT_Stage_Vaneless =[]  

            #List containing variables which will be returned for plotting purposes
            T_diff = [self.gas.T]                                
            tau_diff = [tau_0]  
            
            if self.power_input == 'Temperature': 
                temp_rise=self.T_rise_stage
                    
            elif self.power_input == 'Power': 
                temp_rise=self.power*(1/self.gas.cp_mass)

            for i in range(0, max_stages): 
                if ((round(self.gas.T+temp_rise) not in range(round(self.COT-7), round(self.COT+7))) and (self.gas.T+temp_rise <= self.COT+7)):
                    number += 1 
                    print('Stage:', number)
                    y0 = np.hstack((self.gas.T, self.gas.P, self.gas.Y))                                      

                    # generating residence time lists
                    RT_Stage_Vaneless.append(self.RT_stage*(1-self.Vanes_Multiplier))
                    RT_Stage_Vanes.append(self.RT_stage*self.Vanes_Multiplier) 
                    dt_Vaneless = RT_Stage_Vaneless[i]/ 1e6   
                    dt_Vanes = RT_Stage_Vanes[i]/ 1e6  

                    # non-instanteneous T increase    
                    # Set up objects representing the ODE and the solver; Note that we use here the ODE_Vanes!!                         
                    # This to account for the power addition                          
                    solver = spi.ode(self.outer_instance.ODE_Vanes(self.outer_instance,tau_0, tau_0 + RT_Stage_Vanes[i]/1000))
                    solver.set_integrator('vode', method = 'bdf', with_jacobian = True, order = 5,
                                        atol = solver_atol, rtol = solver_rtol)
                    solver.set_initial_value(y0, tau_0)

                    # Integrate the equations
                    while solver.successful() and solver.t <= (tau_0 + RT_Stage_Vanes[i]/1000)*(1-(1e-6)/100):
                        solver.integrate(solver.t + dt_Vanes)
                        self.gas.TPY = solver.y[0], solver.y[1], solver.y[2:]
                        states.append(self.gas.state, tau=solver.t)                          

                    if not(solver.successful()):
                        print('Simulation was unsuccessful')


                    # Store the tau_0 and y0 as we need this for the consecutive calculation of the Vaneless section.
                    tau_0 = states.tau[-1]    
                    T_outlet.append(self.gas.T)
                    T_diff.append(self.gas.T)                             
                    y0 = np.hstack((self.gas.T, self.gas.P, self.gas.Y)) 
                    tau_diff.append(tau_0)


                     # Set up objects representing the ODE and the solver; Note that we use here the ODE_Vaneless!!                              
                    solver = spi.ode(self.outer_instance.ODE_Vaneless(self.outer_instance))
                    solver.set_integrator('vode', method = 'bdf', with_jacobian = True, order = 5,
                                        atol = solver_atol, rtol = solver_rtol)
                    solver.set_initial_value(y0, tau_0)

                    # Integrate the equations
                    while solver.successful() and solver.t <= (tau_0 + RT_Stage_Vaneless[i]/1000)*(1-(1e-6)/100):
                        solver.integrate(solver.t + dt_Vaneless)
                        self.gas.TPY = solver.y[0], solver.y[1], solver.y[2:]
                        states.append(self.gas.state, tau=solver.t)

                    #storing some variables for plotting purposes
                    tau_0 = states.tau[-1] 
                    T_diff.append(self.gas.T) 
                    for i in range (0,len(T_diff)-2,2): 
                        temp_rise = abs(T_diff[i+1]-T_diff[i])
                        
                    if not(solver.successful()):
                        print('Simulation was unsuccessful') 

                elif (self.gas.T+temp_rise >= self.COT+7): 
                    if (abs((T_outlet[-1]-self.COT)/self.COT) > abs((self.gas.T+temp_rise-self.COT)/self.COT)): 
                        number += 1
                        print('Stage:', number)
                        y0 = np.hstack((self.gas.T, self.gas.P, self.gas.Y))                                      

                        # generating residence time lists
                        RT_Stage_Vaneless.append(self.RT_stage*(1-self.Vanes_Multiplier))
                        RT_Stage_Vanes.append(self.RT_stage*self.Vanes_Multiplier) 
                        dt_Vaneless = RT_Stage_Vaneless[i]/ 1e6   
                        dt_Vanes = RT_Stage_Vanes[i]/ 1e6  

                        # non-instanteneous T increase    
                        # Set up objects representing the ODE and the solver; Note that we use here the ODE_Vanes!!                         
                        # This to account for the power addition                          
                        solver = spi.ode(self.outer_instance.ODE_Vanes(self.outer_instance,tau_0, tau_0 + RT_Stage_Vanes[i]/1000))
                        solver.set_integrator('vode', method = 'bdf', with_jacobian = True, order = 5,
                                            atol = solver_atol, rtol = solver_rtol)
                        solver.set_initial_value(y0, tau_0)

                        # Integrate the equations
                        while solver.successful() and solver.t <= (tau_0 + RT_Stage_Vanes[i]/1000)*(1-(1e-6)/100):
                            solver.integrate(solver.t + dt_Vanes)
                            self.gas.TPY = solver.y[0], solver.y[1], solver.y[2:]
                            states.append(self.gas.state, tau=solver.t)                          

                        if not(solver.successful()):
                            print('Simulation was unsuccessful')

                        # Store the tau_0 and y0 as we need this for the consecutive calculation of the Vaneless section.
                        tau_0 = states.tau[-1]    
                        T_outlet.append(self.gas.T)
                        T_diff.append(self.gas.T)                             
                        y0 = np.hstack((self.gas.T, self.gas.P, self.gas.Y)) 
                        tau_diff.append(tau_0)

                
                         # Set up objects representing the ODE and the solver; Note that we use here the ODE_Vaneless!!                              
                        solver = spi.ode(self.outer_instance.ODE_Vaneless(self.outer_instance))
                        solver.set_integrator('vode', method = 'bdf', with_jacobian = True, order = 5,
                                            atol = solver_atol, rtol = solver_rtol)
                        solver.set_initial_value(y0, tau_0)

                        # Integrate the equations
                        while solver.successful() and solver.t <= (tau_0 + RT_Stage_Vaneless[i]/1000)*(1-(1e-6)/100):
                            solver.integrate(solver.t + dt_Vaneless)
                            self.gas.TPY = solver.y[0], solver.y[1], solver.y[2:]
                            states.append(self.gas.state, tau=solver.t)

                        #storing some variables for plotting purposes
                        tau_0 = states.tau[-1] 
                        T_diff.append(self.gas.T) 
                        for i in range (0,len(T_diff)-2,2): 
                            temp_rise = abs(T_diff[i+1]-T_diff[i])
                            
                        if not(solver.successful()):
                            print('Simulation was unsuccessful')

                    else:
                        RT_TLE = self.RT_stage*(1-self.Vanes_Multiplier)*self.RT_Multiplier-RT_Stage_Vaneless[-1]
                        RT_Stage_Vaneless[-1] = RT_Stage_Vaneless[-1]+RT_TLE
                        dt_TLE = RT_TLE/ 1e6 
                        
                        # Store the tau_0 and y0 as we need this for the consecutive calculation of the region to the TLE.
                        tau_0 = states.tau[-1]    
                        y0 = np.hstack((self.gas.T, self.gas.P, self.gas.Y)) 

                        # Set up objects representing the ODE and the solver; Note that we use here the ODE_Vaneless!!                              
                        solver = spi.ode(self.outer_instance.ODE_Vaneless(self.outer_instance))
                        solver.set_integrator('vode', method = 'bdf', with_jacobian = True, order = 5,
                                            atol = solver_atol, rtol = solver_rtol)
                        solver.set_initial_value(y0, tau_0)
            
            
                        # Integrate the equations
                        while solver.successful() and solver.t <= (tau_0 + RT_TLE/1000)*(1-(1e-6)/100):
                            solver.integrate(solver.t + dt_TLE)
                            self.gas.TPY = solver.y[0], solver.y[1], solver.y[2:]
                            states.append(self.gas.state, tau=solver.t)

                        #storing some variables for plotting purposes
                        tau_0 = states.tau[-1] 
                        T_diff.append(self.gas.T) 

                        if not(solver.successful()):
                            print('Simulation was unsuccessful')


                        break

                # In case the COT is exceeded due to the power addition, we must increase the residence time to account for
                # the time between the reactor and the TLE. This can be simply accounted for by multiplying the time in the 
                # vaneless space by an arbitrary multiplication factor. The latter can be based on CFD or experimental results
                # or used as variable in an optimization routine for finding the best reactor configuration. 
                else:
                    number += 1
                    print('Stage:', number)
                    y0 = np.hstack((self.gas.T, self.gas.P, self.gas.Y))                                      

                    # generating residence time lists
                    RT_Stage_Vaneless.append(self.RT_stage*(1-self.Vanes_Multiplier)* self.RT_Multiplier)
                    RT_Stage_Vanes.append(self.RT_stage*self.Vanes_Multiplier) 
                    dt_Vaneless = RT_Stage_Vaneless[i]/ 1e7   
                    dt_Vanes = RT_Stage_Vanes[i]/ 1e6  

                    # non-instanteneous T increase    
                    # Set up objects representing the ODE and the solver; Note that we use here the ODE_Vanes!!                         
                    # This to account for the power addition                          
                    solver = spi.ode(self.outer_instance.ODE_Vanes(self.outer_instance,tau_0, tau_0 + RT_Stage_Vanes[i]/1000))
                    solver.set_integrator('vode', method = 'bdf', with_jacobian = True, order = 5,
                                        atol = solver_atol, rtol = solver_rtol)
                    solver.set_initial_value(y0, tau_0)

                    # Integrate the equations
                    while solver.successful() and solver.t <= (tau_0 + RT_Stage_Vanes[i]/1000)*(1-(1e-6)/100):
                        solver.integrate(solver.t + dt_Vanes)
                        self.gas.TPY = solver.y[0], solver.y[1], solver.y[2:]
                        states.append(self.gas.state, tau=solver.t)                          

                    if not(solver.successful()):
                        print('Simulation was unsuccessful')

                    # Store the tau_0 and y0 as we need this for the consecutive calculation of the Vaneless section.
                    tau_0 = states.tau[-1]    
                    T_outlet.append(self.gas.T)
                    T_diff.append(self.gas.T)                             
                    y0 = np.hstack((self.gas.T, self.gas.P, self.gas.Y)) 
                    
                     # Set up objects representing the ODE and the solver; Note that we use here the ODE_Vaneless!!                              
                    solver = spi.ode(self.outer_instance.ODE_Vaneless(self.outer_instance))
                    solver.set_integrator('vode', method = 'bdf', with_jacobian = True, order = 5,
                                        atol = solver_atol, rtol = solver_rtol)
                    solver.set_initial_value(y0, tau_0)

                    # Integrate the equations
                    while solver.successful() and solver.t <= (tau_0 + RT_Stage_Vaneless[i]/1000)*(1-(1e-6)/100):
                        solver.integrate(solver.t + dt_Vaneless)
                        self.gas.TPY = solver.y[0], solver.y[1], solver.y[2:]
                        states.append(self.gas.state, tau=solver.t)

                    #storing some variables for plotting purposes
                    tau_0 = states.tau[-1] 
                    T_diff.append(self.gas.T) 

                    if not(solver.successful()):
                        print('Simulation was unsuccessful')

                    break
            total_res_time = sum(RT_Stage_Vaneless)+sum(RT_Stage_Vanes)
            res_time_final_stage =  RT_Stage_Vaneless[-1]+RT_Stage_Vanes[-1]
            return states, number, T_diff,tau_diff

#------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------
#   electrified Rotor-Stator Reactor/ Rotodynamic Reactor (eRSR/RDR) - BASE MODEL, custom reaction rates 
#   Model originally developed by Mike Bonheure (LCT). 
#------------------------------------------------------------------------------------------------------------------------

class customESC_BM:
    """
    Class to solve electric steam cracking reactors which are based on turbomachinery 
    (see for  example,  https://coolbrook.com/). This implementation relies on a 
    user-defined function to calculate the reaction rates, given the thermodynamic state of the gas phase.

    Modeling assumptions
    ----------
    1) Vaneless space is assumed to be an ideal plug flow reactor. Depending on the type of reactor, this assumption is not 
       entirely true (i.e., occurence of lateral mixing between stream tubes of different age). 
    2) The reactor operates ISOBARIC (i.e., the static pressure is assumed to be constant throughout the reactor). 
       This is justified based on the fact that these type of machines are design as such that the pressure increase 
       (which is traditionally seen in a compressor) is nullified by introducing losses (i.e. entropy increase). Hence, accross
       the entire reactor (i.e. inlet - outlet) the pressure will hardly change. 
       
       Of course it must be acknowledged that locally static pressure changes occur but these are not accounted for in the 
       base-model. 
       
    3) The reactor operates ADIABATIC (i.e., there is no heat transfer to the environment).
    4) The code uses PDE's in function of residence time instead of axial position (i.e., like traditional PFR PDE's). 
       This because this approach allows to develop an general applicable code for all turbomachinery-based electric 
       steam crackers in which axial location not always makes physically sense (e.g. toroidal design of coolbrook). 
       
       NOTE: this does not make the equations transient in nature. The residence-time is merely a variable just like axial 
       location. A truly transient PFR, would require PDE's insteady of ODE's, which is not the case here as for 
       the derivations of the modeling equations it is relied upon the steady-state reynolds transport theorem.
    5) The power input per stage is modeled as a simple temperature increase, reduced by the occuring reactions.
       This value can be chosen ad-hoc or based on CFD or experimental results. 
    6) The coil outlet temperature is used as parameter to track the cracking severity. This method allows to determine
       how many stages must be implemented to reach the targeted severity. 
    7) The base model assumes that the cross sectional area does not change, as a consequence you must consider -for the
       physical interpretation- that an indentical residence time in each vaneless space implies that the length of the 
       vaneless space increases as a consequence of the cracking reactions and the reactor operation.

    Parameters
    ----------
    mechanism_file : string
        Name of Cantera input file (*.cti, *.yaml)
    gas_id : string
        Name of the gas phase in the mechanism file
    reaction_rates : callable
        Function `fun(phase, epsB, av, epsC) -> ndarray` to calculate the net
        production rates, given the thermodynamic state of the gas `phase`, the
        bed porosity, the area-per-volume and catalyst porosity.\n
        Returns an array with the net production rate of all species.
    COT : float
        Coil Outlet Temperature [K] as measure for cracking severity\n
    RT_stage: float
        Residence time per stage [ms]\n
        Stage is defined here as both the vaned as vaneless section
    Vanes_Multiplier: float
        Average relative time which the fluid resides within the vaned section [/]\n
        Expressed in relative terms of the total RT_Stage\n
        (e.g. Vanes_Multiplier = 0.1, implies 10% of the total RT_stage is within the vaned section)
    RT_Multiplier: float
        Value with which the residence time in the vaneless section will be multiplied in the last section [/]\n
        This allows to account for the transport from the reactor to the TLE.   
    power_input : string, optional   
        Type of power addition to mimic rotor: 'Temperature'(default), 'Power'\n
    power: float, optional
        (total) enthalpy increase (see Euler's Turbomachine Equation)\n
        Required when power_input = `Power`
    T_rise_stage : float, optional 
        Static temperature increase in vaned section \n
        Required when power_input = `Temperature`
    reaction_vanes :   string, optional 
        Activating the occurence of reactions in the vaned section: 'True' (default), 'False'
    energy_type : string, optional
        Type of energy treatment: `adiabatic` (default),`heat-transfer-surroundings`
    U : float, optional
        Global heat transfer coefficient [W/m2/K]\n
        Required when energy_type = `heat-transfer-surroundings`
    Tr : float, optional
        Temperature of surroundings [K]\n
        Required when energy_type = `heat-transfer-surroundings`
    diam : float, optional
        Shroud diameter of the reactor [m]
        Required when energy_type = `heat-transfer-surroundings`

    Examples
    --------
    >>> def rates(phase):
            rr = np.zeros(phase.n_species)
            rr[phase.species_index('CH4')] =
            return rr
            
    >>> # Operating conditions
    >>>     T_in = 600.0 # temperature [K]
    >>>     P_in = 2.0e5 # pressure [Pa]
    >>>     Y_in = {'C2H6':1.0, 'H2O':0.326}  # mass fractions
    
    >>> # Parameter setting
    >>>     COT = 1000
    >>>     RT_stage = 2
    >>>     Vanes_Multiplier = 0.1 
    >>>     RT_Multiplier = 1.5
    >>>     T_rise_stage = 60

    >>> # Create PFR object and solve - without pressure drop
    >>>     propane_ersr_noDP = customESC_BM(reaction_mechanism, gas_id, rates, COT, RT_stage,Vanes_Multiplier, RT_Multiplier,
                                     power_input  = 'Temperature', T_rise_stage=T_rise_stage)
    >>>     propane_ersr_noDP.set_inlet_TPY(T_in, P_in, Y_in)
    >>>     states_noDP_ersr = propane_ersr_noDP.solve(propane_ersr_noDP)
    >>>     states_noDP_ersr=states_noDP_ersr()

    """

    def __init__(self, mechanism_file, gas_id, reaction_rates, COT, RT_stage, Vanes_Multiplier, RT_Multiplier,
                 power_input = 'Temperature', power = None, T_rise_stage = None, reaction_vanes = True, 
                 energy_type = 'adiabatic', diam = None, U = None, Tr = None):
        
        
        self.gas = ct.Solution(mechanism_file, gas_id)
        self.rr = reaction_rates
        self.reaction_vanes = reaction_vanes
        self.RT_stage = RT_stage
        self.Vanes_Multiplier = Vanes_Multiplier
        self.RT_Multiplier = RT_Multiplier
        self.COT = COT
        
        self.energy_type = energy_type
        if self.energy_type == 'heat-transfer-surroundings':
            if U is None:
                raise Exception('Heat transfer coefficient (U) not specified')
            elif diam is None: 
                raise Exception('Diameter (diam) not specified')
            elif Tr is None:
                raise Exception('Temperature surroundings (Tr) not specified')
            self.U = U
            self.Tr = Tr
            self.diam = diam
            self.P = np.pi*diam
            
        self.power_input = power_input
        if self.power_input == 'Temperature': 
            if T_rise_stage is None:
                raise Exception('Temperature increase due to vaned section (T_rise_stage) not specified')
            self.T_rise_stage = T_rise_stage
        elif self.power_input == 'Power': 
            if power is None:
                raise Exception('Power from rotor to fluid (power) not specified')
            self.power = power
        else: 
            raise Exception('The specified property of power_input is incorrect! Please correct the input.')

    def set_state(self, y):
        """
        Set the state vector
            \\vec{y} = \\{T, p, \\vec{Y_k}\\}
        """

        self.gas.set_unnormalized_mass_fractions(y[2:])
        self.gas.TP = y[0], y[1]

    def set_inlet_TPX(self, T, P, X):
        self.gas.TPX = T, P, X

    def set_inlet_TPY(self, T, P, Y):
        self.gas.TPY = T, P, Y                

    
    # work with nested classes to make sure that the correct solver is used each time one is using this
    class ODE_Vanes: 
        def __init__(self, outer_instance, tau_0, tau):
            self.outer_instance = outer_instance
            
            self.gas = self.outer_instance.gas
            self.reaction_vanes = self.outer_instance.reaction_vanes
            self.rr = self.outer_instance.rr
            self.RT_stage = self.outer_instance.RT_stage
            self.Vanes_Multiplier = self.outer_instance.Vanes_Multiplier
            self.RT_Multiplier = self.outer_instance.RT_Multiplier
            self.COT = self.outer_instance.COT
            self.energy_type = self.outer_instance.energy_type
            if self.energy_type == 'heat-transfer-surroundings':
                self.U = self.outer_instance.U
                self.Tr = self.outer_instance.Tr
                self.diam = self.outer_instance.diam
                self.P = self.outer_instance.P
            
            self.power_input = self.outer_instance.power_input
            if self.power_input == 'Temperature': 
                self.T_rise_stage = self.outer_instance.T_rise_stage
            if self.power_input == 'Power': 
                self.power = self.outer_instance.power
            
            self.tau_ini = tau_0
            self.tau = tau
            
        def set_state(self, y):
            """
            Set the state vector
                \\vec{y} = \\{T, p, \\vec{Y_k}\\}
            """

            self.gas.set_unnormalized_mass_fractions(y[2:])
            self.gas.TP = y[0], y[1]

        def set_inlet_TPX(self, T, P, X):
            self.gas.TPX = T, P, X

        def set_inlet_TPY(self, T, P, Y):
            self.gas.TPY = T, P, Y    
        
        def __call__(self, z, y):  
            """
            The ODE functions for the VANED section of the reactor. 

            Returns
            ------

            .. math::
                \\frac{\\partial \\vec{y}}{\\partial z} = f(z,\\vec{y})

            with

            .. math::
                \\vec{y} = \\{T, p, \\vec{Y_k}\\}

            .. math::
                c_{p,mix}\\dot{m} \\frac{\\partial T}{\\partial z}
                        = - A \\sum_{k=1}^{N_g} h_k(T) R_{k} W_k
                            \\left[+ P q_w\\right] \\left[+ P U (T_r - T)\\right]

            .. math::
                \\frac{\\partial p}{\\partial z} =
                            - \\left( \\mu D + \\frac{1}{2}\\rho F U \\right) U

            .. math::
                \\dot{m} \\frac{\\partial Y_k}{\\partial z} = A R_{k} W_k

            """

            # NOTE: z is the residence time            
            
            self.set_state(y)
            rdot = self.rr(self.gas)
            #Conservation of species
            dYdz = [0]*len(rdot)
            if  self.reaction_vanes == True:
                dYdz = rdot * self.gas.molecular_weights / self.gas.density
            

            #Conservation of Energy
            dTdz = 0.0
            if self.energy_type != 'isothermal':
                if  self.reaction_vanes == True:
                    dTdz = (-np.dot(self.gas.partial_molar_enthalpies, rdot))
                if self.energy_type == 'heat-transfer-surroundings':
                    dTdz += self.U * self.P * (self.Tr - self.gas.T)
                dTdz /= (self.gas.cp_mass*self.gas.density_mass)

            # Additional term on the RHS must be included in the energy equation. This because the design of these reactors, 
            # relies on the transfer of kinetic energy into internal energy (i.e. temperature) through processes like
            # diffusion and loss introduction. 
            
                if self.power_input == 'Temperature': 
                    dTdz += self.T_rise_stage/(self.tau-self.tau_ini)

            # The implementation of Power term is based on the Eulers Turbomachine equation (i.e. Energy conservation) 
            # However, as Euler is an algebraic equation the power input term must be defined as such that it quantifies 
            # a change in states. This can be done by defining the power input as Power/massflow or the total enthalpy difference
            # accross a single stage. We than rely  on the local linear approximation to of the Euler equation to calculate the 
            # slope of the power in function of the residence time. his slope is than divided by the cp value (assuming ideal gas) 
            # to get this in terms of T.   
                elif self.power_input == 'Power': 
                    dTdz +=(1/self.gas.cp_mass)*(self.power/(self.tau-self.tau_ini))                                        
                
            #Conservation of momentum
            dpdz = 0.0

            #Create a single array containing the solutions of all the solved ODE's  
            return np.hstack((dTdz, dpdz, dYdz))
    
    class ODE_Vaneless: 
        def __init__(self, outer_instance):
            self.outer_instance = outer_instance
            
            self.gas = self.outer_instance.gas
            self.reaction_vanes = self.outer_instance.reaction_vanes
            self.rr = self.outer_instance.rr
            self.RT_stage = self.outer_instance.RT_stage
            self.Vanes_Multiplier = self.outer_instance.Vanes_Multiplier
            self.RT_Multiplier = self.outer_instance.RT_Multiplier
            self.COT = self.outer_instance.COT
            self.energy_type = self.outer_instance.energy_type
            if self.energy_type == 'heat-transfer-surroundings':
                self.U = self.outer_instance.U
                self.Tr = self.outer_instance.Tr
                self.diam = self.outer_instance.diam
                self.P = self.outer_instance.P
            
            self.power_input = self.outer_instance.power_input
            if self.power_input == 'Temperature': 
                self.T_rise_stage = self.outer_instance.T_rise_stage
            if self.power_input == 'Power': 
                self.power = self.outer_instance.power

        def set_state(self, y):
            """
            Set the state vector
                \\vec{y} = \\{T, p, \\vec{Y_k}\\}
            """

            self.gas.set_unnormalized_mass_fractions(y[2:])
            self.gas.TP = y[0], y[1]

        def set_inlet_TPX(self, T, P, X):
            self.gas.TPX = T, P, X

        def set_inlet_TPY(self, T, P, Y):
            self.gas.TPY = T, P, Y    
            
        def __call__(self, z, y):  
            
            """
            The ODE functions for the vaned section of the reactor. 

            Returns
            ------

            .. math::
                \\frac{\\partial \\vec{y}}{\\partial z} = f(z,\\vec{y})

            with

            .. math::
                \\vec{y} = \\{T, p, \\vec{Y_k}\\}

            .. math::
                c_{p,mix}\\dot{m} \\frac{\\partial T}{\\partial z}
                        = - A \\sum_{k=1}^{N_g} h_k(T) R_{k} W_k
                            \\left[+ P q_w\\right] \\left[+ P U (T_r - T)\\right]

            .. math::
                \\frac{\\partial p}{\\partial z} =
                            - \\left( \\mu D + \\frac{1}{2}\\rho F U \\right) U

            .. math::
                \\dot{m} \\frac{\\partial Y_k}{\\partial z} = A R_{k} W_k

            """
            # z is the residence time
            
            self.set_state(y)
            rdot = self.rr(self.gas)

            #Conservation of species
            dYdz = rdot * self.gas.molecular_weights / self.gas.density

            #Conservation of Energy
            dTdz = 0.0
            if self.energy_type != 'isothermal':
                dTdz = (-np.dot(self.gas.partial_molar_enthalpies, rdot))
                if self.energy_type == 'heat-transfer-surroundings':
                    dTdz += self.U * self.P * (self.Tr - self.gas.T)
                dTdz /= (self.gas.cp_mass*self.gas.density_mass)

            #Conservation of momentum
            dpdz = 0.0

            #Create a single array containing the solutions of all the solved ODE's                                
            return np.hstack((dTdz, dpdz, dYdz))


    class solve:
        def __init__(self, outer_instance):
            self.outer_instance = outer_instance          
            
            self.gas = self.outer_instance.gas
            self.reaction_vanes = self.outer_instance.reaction_vanes
            self.rr = self.outer_instance.rr
            self.RT_stage = self.outer_instance.RT_stage
            self.Vanes_Multiplier = self.outer_instance.Vanes_Multiplier
            self.RT_Multiplier = self.outer_instance.RT_Multiplier
            self.COT = self.outer_instance.COT
            self.energy_type = self.outer_instance.energy_type
            if self.energy_type == 'heat-transfer-surroundings':
                self.U = self.outer_instance.U
                self.Tr = self.outer_instance.Tr
                self.diam = self.outer_instance.diam
                self.P = self.outer_instance.P
            
            self.power_input = self.outer_instance.power_input
            if self.power_input == 'Temperature': 
                self.T_rise_stage = self.outer_instance.T_rise_stage
            if self.power_input == 'Power': 
                self.power = self.outer_instance.power
        

        def __call__(self,solver_rtol = 1e-9, solver_atol = 1e-15): 

            """
            Solve the reactor model.

            NOTE: the initial state is the current state of the gas phase object,
            which has to be set prior to calling this `solve` function.

            Parameters
            ----------
            solver_rtol : float, optional
                Relative tolerance for ODE solver
            solver_atol : float, optional
                Absolute tolerance for ODE solver

            Returns
            ------
            states : SolutionArray
                Object containing thermodynamic states as a function of position z

            """

            #variable initialization 
            tau_0 = 0.0
            number = 0
            max_stages = 1000 # This value can be changed! 
                              # but of course question yourself if a reactor with +1000 stages is feasible?                                 
            states = ct.SolutionArray(self.gas, 1, extra={'tau': [tau_0]})

            #list declarations used throughout the solving routine
            T_outlet = [self.gas.T]
            RT_Stage_Vanes =[]
            RT_Stage_Vaneless =[]  

            #List containing variables which will be returned for plotting purposes
            T_diff = [self.gas.T]                                
            tau_diff = [tau_0]  
            
            if self.power_input == 'Temperature': 
                temp_rise=self.T_rise_stage
                    
            elif self.power_input == 'Power': 
                temp_rise=self.power*(1/self.gas.cp)

            for i in range(0, max_stages): 
                
                
                if ((round(self.gas.T+temp_rise) not in range(round(self.COT-7), round(self.COT+7))) and (self.gas.T+temp_rise <= self.COT+7)):
                    number += 1 
                    print('Stage:', number)
                    y0 = np.hstack((self.gas.T, self.gas.P, self.gas.Y))                                      

                    # generating residence time lists
                    RT_Stage_Vaneless.append(self.RT_stage*(1-self.Vanes_Multiplier))
                    RT_Stage_Vanes.append(self.RT_stage*self.Vanes_Multiplier) 
                    dt_Vaneless = RT_Stage_Vaneless[i]/ 1e6   
                    dt_Vanes = RT_Stage_Vanes[i]/ 1e6  

                    # non-instanteneous T increase    
                    # Set up objects representing the ODE and the solver; Note that we use here the ODE_Vanes!!                         
                    # This to account for the power addition                          
                    solver = spi.ode(self.outer_instance.ODE_Vanes(self.outer_instance,tau_0, tau_0 + RT_Stage_Vanes[i]/1000))
                    solver.set_integrator('vode', method = 'bdf', with_jacobian = True, order = 5,
                                        atol = solver_atol, rtol = solver_rtol)
                    solver.set_initial_value(y0, tau_0)

                    # Integrate the equations
                    while solver.successful() and solver.t <= (tau_0 + RT_Stage_Vanes[i]/1000)*(1-(1e-6)/100):
                        solver.integrate(solver.t + dt_Vanes)
                        self.gas.TPY = solver.y[0], solver.y[1], solver.y[2:]
                        states.append(self.gas.state, tau=solver.t)                          

                    if not(solver.successful()):
                        print('Simulation was unsuccessful')

                    # Store the tau_0 and y0 as we need this for the consecutive calculation of the Vaneless section.
                    tau_0 = states.tau[-1]    
                    T_outlet.append(self.gas.T)
                    T_diff.append(self.gas.T)                             
                    y0 = np.hstack((self.gas.T, self.gas.P, self.gas.Y)) 
                    tau_diff.append(tau_0)


                     # Set up objects representing the ODE and the solver; Note that we use here the ODE_Vaneless!!                              
                    solver = spi.ode(self.outer_instance.ODE_Vaneless(self.outer_instance))
                    solver.set_integrator('vode', method = 'bdf', with_jacobian = True, order = 5,
                                        atol = solver_atol, rtol = solver_rtol)
                    solver.set_initial_value(y0, tau_0)

                    # Integrate the equations
                    while solver.successful() and solver.t <= (tau_0 + RT_Stage_Vaneless[i]/1000)*(1-(1e-6)/100):
                        solver.integrate(solver.t + dt_Vaneless)
                        self.gas.TPY = solver.y[0], solver.y[1], solver.y[2:]
                        states.append(self.gas.state, tau=solver.t)

                    #storing some variables for plotting purposes
                    tau_0 = states.tau[-1] 
                    T_diff.append(self.gas.T) 
                    for i in range (0,len(T_diff)-2,2): 
                        temp_rise = abs(T_diff[i+1]-T_diff[i])
                        
                    if not(solver.successful()):
                        print('Simulation was unsuccessful') 

                elif (self.gas.T+temp_rise >= self.COT+7): 
                    if (abs((T_outlet[-1]-self.COT)/self.COT) > abs((self.gas.T+temp_rise-self.COT)/self.COT)): 
                        number += 1
                        print('Stage:', number)
                        y0 = np.hstack((self.gas.T, self.gas.P, self.gas.Y))                                      

                        # generating residence time lists
                        RT_Stage_Vaneless.append(self.RT_stage*(1-self.Vanes_Multiplier))
                        RT_Stage_Vanes.append(self.RT_stage*self.Vanes_Multiplier) 
                        dt_Vaneless = RT_Stage_Vaneless[i]/ 1e6   
                        dt_Vanes = RT_Stage_Vanes[i]/ 1e6  

                        # non-instanteneous T increase    
                        # Set up objects representing the ODE and the solver; Note that we use here the ODE_Vanes!!                         
                        # This to account for the power addition                          
                        solver = spi.ode(self.outer_instance.ODE_Vanes(self.outer_instance,tau_0, tau_0 + RT_Stage_Vanes[i]/1000))
                        solver.set_integrator('vode', method = 'bdf', with_jacobian = True, order = 5,
                                            atol = solver_atol, rtol = solver_rtol)
                        solver.set_initial_value(y0, tau_0)

                        # Integrate the equations
                        while solver.successful() and solver.t <= (tau_0 + RT_Stage_Vanes[i]/1000)*(1-(1e-6)/100):
                            solver.integrate(solver.t + dt_Vanes)
                            self.gas.TPY = solver.y[0], solver.y[1], solver.y[2:]
                            states.append(self.gas.state, tau=solver.t)                          

                        if not(solver.successful()):
                            print('Simulation was unsuccessful')

                        # Store the tau_0 and y0 as we need this for the consecutive calculation of the Vaneless section.
                        tau_0 = states.tau[-1]    
                        T_outlet.append(self.gas.T)
                        T_diff.append(self.gas.T)                             
                        y0 = np.hstack((self.gas.T, self.gas.P, self.gas.Y)) 
                        tau_diff.append(tau_0)

                
                         # Set up objects representing the ODE and the solver; Note that we use here the ODE_Vaneless!!                              
                        solver = spi.ode(self.outer_instance.ODE_Vaneless(self.outer_instance))
                        solver.set_integrator('vode', method = 'bdf', with_jacobian = True, order = 5,
                                            atol = solver_atol, rtol = solver_rtol)
                        solver.set_initial_value(y0, tau_0)

                        # Integrate the equations
                        while solver.successful() and solver.t <= (tau_0 + RT_Stage_Vaneless[i]/1000)*(1-(1e-6)/100):
                            solver.integrate(solver.t + dt_Vaneless)
                            self.gas.TPY = solver.y[0], solver.y[1], solver.y[2:]
                            states.append(self.gas.state, tau=solver.t)

                        #storing some variables for plotting purposes
                        tau_0 = states.tau[-1] 
                        T_diff.append(self.gas.T) 
                        for i in range (0,len(T_diff)-2,2): 
                            temp_rise = abs(T_diff[i+1]-T_diff[i])
                            
                        if not(solver.successful()):
                            print('Simulation was unsuccessful')

                    else:
                        RT_TLE = self.RT_stage*(1-self.Vanes_Multiplier)-RT_Stage_Vaneless[-1]
                        RT_Stage_Vaneless[-1] = RT_Stage_Vaneless[-1]+RT_TLE
                        dt_TLE = RT_TLE/ 1e6 
                        
                        # Store the tau_0 and y0 as we need this for the consecutive calculation of the region to the TLE.
                        tau_0 = states.tau[-1]    
                        y0 = np.hstack((self.gas.T, self.gas.P, self.gas.Y)) 

                        # Set up objects representing the ODE and the solver; Note that we use here the ODE_Vaneless!!                              
                        solver = spi.ode(self.outer_instance.ODE_Vaneless(self.outer_instance))
                        solver.set_integrator('vode', method = 'bdf', with_jacobian = True, order = 5,
                                            atol = solver_atol, rtol = solver_rtol)
                        solver.set_initial_value(y0, tau_0)
            
            
                        # Integrate the equations
                        while solver.successful() and solver.t <= (tau_0 + RT_TLE/1000)*(1-(1e-6)/100):
                            solver.integrate(solver.t + dt_TLE)
                            self.gas.TPY = solver.y[0], solver.y[1], solver.y[2:]
                            states.append(self.gas.state, tau=solver.t)

                        #storing some variables for plotting purposes
                        tau_0 = states.tau[-1] 
                        T_diff.append(self.gas.T) 

                        if not(solver.successful()):
                            print('Simulation was unsuccessful')
            
                # In case the COT is exceeded due to the power addition, we must increase the residence time to account for
                # the time between the reactor and the TLE. This can be simply accounted for by multiplying the time in the 
                # vaneless space by an arbitrary multiplication factor. The latter can be based on CFD or experimental results
                # or used as variable in an optimization routine for finding the best reactor configuration. 
                else:
                    number += 1
                    print('Stage:', number)
                    y0 = np.hstack((self.gas.T, self.gas.P, self.gas.Y))                                      

                    # generating residence time lists
                    RT_Stage_Vaneless.append(self.RT_stage*(1-self.Vanes_Multiplier)* self.RT_Multiplier)
                    RT_Stage_Vanes.append(self.RT_stage*self.Vanes_Multiplier) 
                    dt_Vaneless = RT_Stage_Vaneless[i]/ 1e7   
                    dt_Vanes = RT_Stage_Vanes[i]/ 1e6  

                    # non-instanteneous T increase    
                    # Set up objects representing the ODE and the solver; Note that we use here the ODE_Vanes!!                         
                    # This to account for the power addition                          
                    solver = spi.ode(self.outer_instance.ODE_Vanes(self.outer_instance,tau_0, tau_0 + RT_Stage_Vanes[i]/1000))
                    solver.set_integrator('vode', method = 'bdf', with_jacobian = True, order = 5,
                                        atol = solver_atol, rtol = solver_rtol)
                    solver.set_initial_value(y0, tau_0)

                    # Integrate the equations
                    while solver.successful() and solver.t <= (tau_0 + RT_Stage_Vanes[i]/1000)*(1-(1e-6)/100):
                        solver.integrate(solver.t + dt_Vanes)
                        self.gas.TPY = solver.y[0], solver.y[1], solver.y[2:]
                        states.append(self.gas.state, tau=solver.t)                          

                    if not(solver.successful()):
                        print('Simulation was unsuccessful')

                    # Store the tau_0 and y0 as we need this for the consecutive calculation of the Vaneless section.
                    tau_0 = states.tau[-1]    
                    T_outlet.append(self.gas.T)
                    T_diff.append(self.gas.T)                             
                    y0 = np.hstack((self.gas.T, self.gas.P, self.gas.Y)) 
                    
                     # Set up objects representing the ODE and the solver; Note that we use here the ODE_Vaneless!!                              
                    solver = spi.ode(self.outer_instance.ODE_Vaneless(self.outer_instance))
                    solver.set_integrator('vode', method = 'bdf', with_jacobian = True, order = 5,
                                        atol = solver_atol, rtol = solver_rtol)
                    solver.set_initial_value(y0, tau_0)

                    # Integrate the equations
                    while solver.successful() and solver.t <= (tau_0 + RT_Stage_Vaneless[i]/1000)*(1-(1e-6)/100):
                        solver.integrate(solver.t + dt_Vaneless)
                        self.gas.TPY = solver.y[0], solver.y[1], solver.y[2:]
                        states.append(self.gas.state, tau=solver.t)

                    #storing some variables for plotting purposes
                    tau_0 = states.tau[-1] 
                    T_diff.append(self.gas.T) 

                    if not(solver.successful()):
                        print('Simulation was unsuccessful')

                    break
            total_res_time = sum(RT_Stage_Vaneless)+sum(RT_Stage_Vanes)
            res_time_final_stage =  RT_Stage_Vaneless[-1]+RT_Stage_Vanes[-1]
            return states, number, T_diff,tau_diff

#------------------------------------------------------------------------------------------------------------------------