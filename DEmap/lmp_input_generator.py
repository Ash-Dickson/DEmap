
def generate_lammps_input(
    PKA_id=None,
    units=None,
    atom_style=None,
    read_data=None,
    masses=None,
    pair_styles=None,
    pair_coeffs=None,
    run_steps=None
):
    """
    Generates a LAMMPS input file (tde.in) from a set of parameters.

    Parameters:
        PKA_id (int): ID of the PKA atom
        units (str): Units for the simulation (default: 'metal')
        atom_style (str): Atom style for the simulation (default: 'atomic')
        read_data (str): Filename of the data file to read in (default: 'structure.data')
        masses (list of tuples): List of tuples containing the atom type and mass for each atom
        pair_styles (list of str): List of pair styles to use in the simulation
        pair_coeffs (list of str): List of pair coefficients to use in the simulation
        run_steps (int): Number of steps to run in the simulation (default: 2000)

    Returns:
        None
    """
    if masses is None:
        print('ERROR: Must specify masses for atoms (list of tuples [(atom_type, mass),...])')
        exit()
    if pair_styles is None:
        print('ERROR: Must specify pair styles (list of strings)')
        exit()
    if pair_coeffs is None:
        print('ERROR: Must specify pair coeffs (list of strings)')
        exit()
    if PKA_id is None:
        print('ERROR: Must specify PKA id (int)')
        exit()
    if read_data is None:
        print('ERROR: Must specify data file (str)')
        exit()
    if run_steps is None:   
        print('ERROR: Must specify run steps (int)')
        exit()
    
    if units is None:
        print(f"ERROR: Must specify units (str)")
        exit()

    if atom_style is None:
        print(f"ERROR: Must specify atom_style (str)")
        exit()

    lines = []
    # Basic setup
    lines.append(f"units {units}")
    lines.append(f"dimension 3")
    lines.append(f"boundary p p p")
    lines.append("")
    lines.append(f"atom_style {atom_style}")
    lines.append("")
    lines.append(f"read_data ../{read_data}")
    lines.append("")

    # Masses
    for atom_type, mass in masses:
        lines.append(f"mass {atom_type} {mass}")
    lines.append("")

    # Pair styles
    for ps in pair_styles:
        lines.append(f"{ps}")
    lines.append("")
    for pc in pair_coeffs:
        lines.append(f"{pc}")
    lines.append("")

    lines.extend([
        "neigh_modify delay 0 every 1 check yes",
        "",
        f"thermo_style custom step temp pe ke press vol cella cellb cellc",
        f"thermo 10",
        ""
    ])

    # Box and outer shell
    lines.extend([
        "# Define the entire simulation box dimensions",
        "variable xlo equal bound(all,xmin)",
        "variable xhi equal bound(all,xmax)",
        "variable ylo equal bound(all,ymin)",
        "variable yhi equal bound(all,ymax)",
        "variable zlo equal bound(all,zmin)",
        "variable zhi equal bound(all,zmax)",
        "",
        "# Define variables for the outer shell boundaries",
        "variable xlo_outer equal ${xlo} + 2.5",
        "variable xhi_outer equal ${xhi} - 2.5",
        "variable ylo_outer equal ${ylo} + 2.5",
        "variable yhi_outer equal ${yhi} - 2.5",
        "variable zlo_outer equal ${zlo} + 2.5",
        "variable zhi_outer equal ${zhi} - 2.5",
        "",
        "# Define regions for the outer shell",
        "region outer_xlo block ${xlo} ${xlo_outer} INF INF INF INF",
        "region outer_xhi block ${xhi_outer} ${xhi} INF INF INF INF",
        "region outer_ylo block INF INF ${ylo} ${ylo_outer} INF INF",
        "region outer_yhi block INF INF ${yhi_outer} ${yhi} INF INF",
        "region outer_zlo block INF INF INF INF ${zlo} ${zlo_outer}",
        "region outer_zhi block INF INF INF INF ${zhi_outer} ${zhi}",
        "",
        "# Combine the regions for the outer shell",
        "region outer_shell union 6 outer_xlo outer_xhi outer_ylo outer_yhi outer_zlo outer_zhi",
        "",
        "# Group atoms in the outer shell",
        "group boundary region outer_shell",
        "",
        "group core subtract all boundary",
        f"group PKA id {PKA_id}",
        "",
        f"velocity PKA set MYPKAVEL",
        "fix core_fix all nve",
        "fix boundary_fix boundary langevin 0 0 0.1 48279",
        f"dump positions all atom {run_steps} prod.data",
        "dump_modify positions first yes",
        "",
        f"run {run_steps}"
    ])

    # Write to file
    with open('tde.in', "w") as f:
        f.write("\n".join(lines))

    print(f"LAMMPS input file 'tde.in' generated successfully!")

