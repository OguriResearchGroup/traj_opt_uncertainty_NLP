# Trajectory Optimization Under Uncertainty with Forward Backward Shooting and Nonlinear Programming
## Description
A nonlinear programming approach to trajectory optimization under Gaussian uncertainty using SNOPT. The implementation supports orbit-to-orbit or point-to-point transfers in the two-body problem and three-body problem with initial state Gaussian dispersion, stochastic acceleration, and control execution error. This implementation incorporates feedback gain parametrization, which greatly reduces the number of free variables traditionally associated with linear feedback policy optimization for covariance steering, and gravity limiting, which prevents numerical integration failure due to point mass singularities in the equations of motion. Linear covariance analysis is computed from symbolically defeind equations of motion; no anlytical work is required to use new dynamics models other than definition of equations of motion.

## Instructions for Use
To install the necessary Python environment used to run this code, see the Installation Guide. 

### Configuration Files
The code is driven by YAML configuration files. Several example cases have been provided under ```./configurations```. These files have different sections, which are detailed below.

- name - ```string```: name of the case, does not impact code
- dynamics
    - type - ```string``` ("2BP" or "CR3BP"): dynamics regime. Currently, only the two body problem and circular restricted three body problem are supported
    - mass_rat - ```float```:  if dynamics are "CR3BP", the mass ratio $\mu$ that determines gravitational influence of CR3BP (non-dimensional units)
    - mu1 - ```float```:  if dynamics are "CR3BP", the gravitational parameter of the first primary body (km^3/s^2)
    - mu2 - ```float```:  if dynamics are "CR3BP", the gravitational parameter of the second primary body (km^3/s^2)
    - mu - ```float```:  if dynamics are "2BP", the gravitational parameter of the central body (km^3/s^2)
    - l_star - ```float```: characteristic length used to nondimensionalize
- integration
    - a_tol - ```float```: absolute tolerance of the numerical integrator
    - r_tol - ```float```: relative tolerance of the numerical integrator
    - int_points - ```int```: number of integration points per segment that are saved
    - mc_div - ```int```: nuumber of additional divisions past int_points used to propagate equal time-spaced Monte Carlo trials
- segments - ```int```: number of segment boundaries
- boundary_conditions
    - t0 - ```string``` ("YYYY-MM-DD HH:MM:SS"): epoch of departure
    - tf - ```string``` ("YYYY-MM-DD HH:MM:SS"): epoch of arrival
    - y0 - ```array<float>```: initial state or initial state that can be used to propagate departure orbit (non-dimensional units)   
    - yf - ```array<float>```: final state or initial state that can be used to propagate arrival orbit (non-dimensional units)
    - type - ```string``` ("free" or "fixed"): type of phasing. "free" allows orbit-orbit transfers, whereas "fixed" allows point-to-point
    - per1 - ```float```: if type is "free", the period of the initial orbit
    - per1 - ```float```: if type is "free", the period of the final orbit
    - alpha: 
        - min - ```float```: if type is "free", minimum phasing parameter of the departure orbit [0, 1]
        - max - ```float```: if type is "free", maximum phasing parameter of the departure orbit [0, 1]
    - beta: 
        - min - ```float```: if type is "free", minimum phasing parameter of the arrival orbit [0, 1]
        - max - ```float```: if type is "free", maximum phasing parameter of the arrival orbit [0, 1]
- engine:
    - m0 - ```float```: initial spacecraft mass (kg)
    - Isp - ```float```: specific impulse (s)
    - T_max - ```float```: maximum thrust (N)
- uncertainty:
    - covariance:
        - initial:
            - pos_sig - ```float```: 1 standard deviation for the initial position uncertainty (km)
            - vel_sig - ```float```: 1 standard deviation for the initial velocity uncertainty (km/s)
            - mass_sig - ```float```: 1 standard deviation for the initial mass uncertainty (kg). Keep this very low as we assume essentially zero uncertainty in mass
        - final:
            - pos_sig - ```float```: 1 standard deviation for the final position uncertainty (km)
            - vel_sig - ```float```: 1 standard deviation for the final velocity uncertainty (km/s)
            - mass_sig - ```float```: 1 standard deviation for the final mass uncertainty (kg). Keep this very high as we do not wish to control the mass distribution
    - acc_sig - ```float```: 1 standard deviation for stochastic acceleration (km/s^2)
    - gates
        - fixed_mag - ```float```: 1 standard deviation for Gates model fixed magnitude uncertainty (%)
        - prop_mag - ```float```: 1 standard deviation for Gates model proportional magnitude uncertainty (%)
        - fixed_point - ```float```: 1 standard deviation for Gates model fixed pointing uncertainty (%)
        - prop_point - ```float```: 1 standard deviation for Gates model proportional pointing uncertainty (%)
    - eps - ```float```: fraction of allowed violation of chance constraint, e.g. eps = .001 means 99.9% statistical constraint satisfaction
- UT
    - alpha - ```float```: tunable parameter of scaled unscented transform
    - beta - ```float```: tunable parameter of scaled unscented transform
    - kappa - ```float```: tunable parameter of scaled unscented transform
- SNOPT
    - major_opt_tol - ```float```: major optimality tolerance of SNOPT
    - major_feas_tol - ```float```: major feasibility tolerance of SNOPT
    - minor_feas_tol - ```float```: minor feasibility tolerance of SNOPT
    - partial_price - ```float```: partial pricing of SNOPT
    - linesearch_tol - ```float```: linesearch tolerance of SNOPT
    - function_prec - ```float```: function precision of objective/constraint functions
- constraints
    - deterministic
        - det_col_avoid
            - bool - ```bool``` (true or false): whether deterministic collision avoidance constraint is turned on
            - parameters:
                - r_obs - ```array<float>```: position of obstacle/center of keep-out zone (km)
                - d_safe - ```float```: safe distance of keep-out zone (km)
    - stochastic
        - det_col_avoid
            - bool - ```bool``` (true or false): whether deterministic collision avoidance constraint is turned on
            - parameters:
                - r_obs - ```array<float>```: position of obstacle/center of keep-out zone (km)
                - d_safe - ```float```: safe distance of keep-out zone (km)
        - stat_col_avoid
            - bool - ```bool``` (true or false): whether statistical collision avoidance constraint is turned on
            - parameters:
                - r_obs - ```array<float>```: position of obstacle/center of keep-out zone (km)
                - d_safe - ```float```: safe distance of keep-out zone  (km)

# Installation Guide
This guide will explain how to install ```PyOptSparse``` with the necessary binding to SNOPT7. This guide is for Windows.
However, sections done in WSL could be done in MacOS or Linux. This guide assumes mild familiarity with the Python distribution manager Anaconda.

Prepared by Jerry Varghese on 8/14/2023
## Changelog
- 8/14/2023 - document created
- 5/1/2025  - document updated with additional necessary packages

## WSL2/Ubuntu
1. Install WSL2 by running ```wsl --install``` as administrator in Windows Command Prompt. Refer to https://learn.microsoft.com/en-us/windows/wsl/install for more information
2. Install Ubuntu from the Microsoft Store. **Note: this guide is known to work with Ubuntu 20.04.5 LTS**
3. Follow the instructions given in the prompt to set up a username and password for the virtual machine. Keep this terminal window open; this window is an Ubuntu Terminal and will be referred to as such henceforth.

## Anaconda
1. From any Windows Browser, navigate to ```https://www.anaconda.com/download``` and click on the Linux installer (the small penguin icon underneath the main install button).
2. Move the resulting ```.sh``` file from Downloads to the Ubuntu home directory.
   - Within the file explorer, a new entry should now be visible beneath ```This PC``` and ```Network``` that says Linux. The files associated with the WSL2 virtual machines live here.
   - Under Linux, ```Ubuntu-XX.XX``` should be visible. Navigate to ```home/your-user-name```. Move the Anaconda installer to this location.
   - You may also use ```wget``` from the command line to download the installer.
5. Also move the SNOPT7 folder to the ```home/your-user-name``` location.
6. In the Ubuntu Terminal, run the following command to install all dependencies of Anaconda:
```apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6```
7. Ensure that the installer is in your home directory by running ```ls``` and verifying that the installer is listed.
8. Run ```chmod +x your-installer.sh``` to give the installer executable status.
9. Run ```./your-installer.sh``` to start installation. Follow the prompts in the terminal.
   - The first step is accepting the EULA. It is easy to accidentally hold enter and run all the way through the EULA which will default to selecting ```no``` to whether the terms are accepted. Stop holding enter at ```Cryptography Notice``` and press enter individually until the ```do you accept``` prompt is reached.
   - Anaconda should be installed in the default location that the installer suggests.
   - Type ```yes``` when the installer asks whether conda should be initialized.
10. Close and reopen the Ubuntu terminal.

## Virtual Environment
1. In Ubuntu Terminal, run ```conda create -n snopt_lin python=3.7```. ```snopt_lin``` can be replaced with any name that is desired. This creates a python environment that will be specifically dedicated to installing the packages needed to run PyOptSparse. **Note that this guide is known to work with python=3.7, and may not work for other versions**
2. Run ```conda activate snopt_lin``` to activate the environment.
3. Run ```conda config --add channels conda-forge``` to add conda-forge to the available channels.
4. Run ```conda config --set channel_priority strict```

## ```build_pyoptsparse```
1. Create a ```repos```  folder in ```home/your-user-name```. 
2. Run ```cd repos``` to move working directories.
3. Run ```git clone https://github.com/OpenMDAO/build_pyoptsparse.git``` to download the ```build_pyoptsparse``` package.
4. Run ```python -m pip install ./build_pyoptsparse``` to install the library
5. Run ```sudo apt install swig python3 python3-pip libblas-dev liblapack-dev```
8. Close the Ubuntu terminal and reopen it.
9. Run ```build_pyoptsparse -s ../snopt7```
10. The installer will recommend you run a specific command to set the environment variable; do so.

## Other necessary packages
1. Run ```conda install jax numpy scipy matplotlib astropy```
2. Run ```python -m pip install diffrax``` 