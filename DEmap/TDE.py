import numpy as np
from ovito.io import *
from ovito.modifiers import *
from ovito.pipeline import *
import numpy as np
import subprocess
import random
import os
from DEmap.config import Config
from typing import Tuple, Optional, Callable, List

class TDE_simulation:

    def __init__(self, direction=None, run_line=None):

        # direction is np array 
        """
        Initialises a TDE_simulation object.

        Parameters:
            direction (np.ndarray): Direction of the TDE simulation
            run_line (str): LAMMPS command to run the simulation

        Attributes:
            direction (np.ndarray): Direction of the TDE simulation
            mass_data (dict): Masses of each atom type
            atom_id (int): ID of the PKA atom
            lammps_data_file (str): Filename of the LAMMPS data file
            pka_mass (float): Mass of the PKA atom
            run_line (str): LAMMPS command to run the simulation
        """
        self.direction = direction

        mass_data  = {}
        with open('tde.in', 'r') as f:
            lines = f.readlines()
        for line in lines:
            if 'mass' in line:
                _, atom_type, mass = line.split()
                mass_data[int(atom_type)] = float(mass)
            if 'PKA id' in line:
                atom_id = int(line.split()[-1])
            if 'read_data' in line:
                self.lammps_data_file = line.split()[-1]
 

        self.mass_data = mass_data
        

        # ID of PKA atom
        self.atom_id = atom_id
        if not isinstance(self.atom_id, int):
            raise TypeError('atom_id must be an integer')
        

        # Initialise PKA mass at start of calcualtions
        self.get_pka_mass()

        self.run_line = run_line




    def get_pka_mass (self):
        """
        Retrieves the mass of the PKA atom from the LAMMPS data file.
        """
        with open (self.lammps_data_file.strip('../'), 'r') as f:
            lines = f.readlines()
        for index, line in enumerate(lines):
            if 'Atoms' in line:
                atom_data_start = index + 2
        data_lines = lines[atom_data_start:]

        # Parse data for PKA atom
        for line in data_lines:
            atom_data = line.split()
            if int(atom_data[0]) == int(self.atom_id):
                atom_type = atom_data[1]
                break
    
        self.pka_mass =  self.mass_data[int(atom_type)]



    def calculate_pka_vel (self, pka_vector, pka_energy):
        """
        Calculates the velocity vector of the PKA atom given its direction and energy.

        Parameters:
            pka_vector (list of floats): direction of the PKA atom
            pka_energy (float): energy of the PKA atom in eV

        Returns:
            pka_velocity_vector (list of floats): velocity vector of the PKA atom in lammps units
        """
        eV_to_J = 1.60218e-19  # J/eV
        amu_to_kg = 1.66e-27   # kg/amu

        pka_vector_magnitude = np.linalg.norm(pka_vector)
        unit_vector = pka_vector / pka_vector_magnitude


        pka_velocity = np.sqrt(2 * pka_energy * eV_to_J / (self.pka_mass * amu_to_kg))
        pka_velocity = pka_velocity/100 #in lammps units

        pka_velocity_vector = pka_velocity * unit_vector

        # print(f"Velocity magnitude: {pka_velocity:.2f} m/s")
        # print(f"Velocity vector: {pka_velocity_vector}")

        # pka velocity vector is list of floats
        return pka_velocity_vector

    def defect_check (self):

        # print('Checking for defects..')
        """
        Checks the simulation for defects.

        Returns:
            int: The total number of defects (antisite defects + vacancies + interstitial defects)
        """
        filename = 'prod.data'
        pipeline = import_file(filename)


        ws_modifier = WignerSeitzAnalysisModifier(
            per_type_occupancies = True
            #eliminate_cell_deformation = True,
            #affine_mapping = ReferenceConfigurationModifier.AffineMapping.ToReference
        )


        pipeline.modifiers.append(ws_modifier)
        for frame in range(1, pipeline.source.num_frames):
            data = pipeline.compute(frame)
            occupancies = data.particles['Occupancy'].array
            occupancy2 = 0 #total num interstitial
            occupancy0 = 0 #total num vacancies
            # Get the site types as additional input:
            site_type = data.particles['Particle Type'].array
            # Calculate total occupancy of every site:
            try:
                total_occupancy = np.sum(occupancies, axis=1)
            except np.AxisError:
                total_occupancy = occupancies
            #print(total_occupancy)
            for element in total_occupancy:
                if element == 0:
                    occupancy0 +=1
                if element >= 2:
                    occupancy2 += (1 + (element-2))
            # Set up a particle selection by creating the Selection property:
            selection = data.particles_.create_property('Selection')

            # total number of types
            type_list = [x for x in range(1, len(self.mass_data)+1)]
            # This logic should work generall for any number of types
            # Lower triangular matrix of pairs excluding diagonal contains all unique pairs
            pair_matrix = np.array([[[i, j] for j in type_list] for i in type_list])
            i_lower_ex, j_lower_ex = np.tril_indices(len(type_list), k=-1)
            lower_triangular_matrix = pair_matrix[i_lower_ex, j_lower_ex] # Unique pairs

            for n in range(len(lower_triangular_matrix)):
                # Select A-sites occupied by exactly one B, C, or D atom
                # (the occupancy of the corresponding atom type must be 1, and all others 0)
                selection[...] |= ((site_type == lower_triangular_matrix[n][0]) & (occupancies[:, lower_triangular_matrix[n][1] - 1] == 1) & (total_occupancy == 1)) 
                             
                
            antisite_indices = np.where(selection == 1)[0]


            # Count the total number of antisite defects
            antisite_count = np.count_nonzero(selection[...])

            # Output the total number of antisites as a global attribute:
            data.attributes['Antisite_count'] = antisite_count
            tot_num_defects =  antisite_count + occupancy0 + occupancy2

            defect_count = tot_num_defects
        return defect_count

    def modify_lammps_in (self, filename, pka_vector):
        """
        Modifies the LAMMPS input file by setting the velocity of the PKA atom
        and the group ID of the PKA atom.

        Parameters:
            filename (str): The name of the LAMMPS input file to modify
            pka_vector (list of floats): The velocity vector of the PKA atom

        Returns:
            None
        """
        with open (filename, 'r') as f:
            lines = f.readlines()

        with open (filename, 'w') as f:
            for line in lines:
                if 'velocity PKA set' in line:
                    f.write(f'velocity PKA set {pka_vector[0]} {pka_vector[1]} {pka_vector[2]}\n')
                elif 'group PKA id' in line:
                    f.write(f'group PKA id {self.atom_id}\n')
                else:
                    f.write(line)


    def run_tde_loop (self, start_energy=None):
        """
        Runs a loop to search for the threshold defect energy (TDE).

        The loop starts at the given start_energy and increments the energy by 2 eV until a defect is found.
        Once a defect is found, the loop steps back down by 1 eV until no defect is found.
        The lowest energy with a defect found is returned as the TDE.

        Parameters:
            start_energy (float): The energy to start the search at (default: 5 eV)

        Returns:
            float: The lowest energy with a defect found (TDE)
        """
        start_energy = start_energy if start_energy is not None else 5
        energy_increment = 2
        max_energy = 1000

        energy = start_energy
        tde_energy = None
        while True:
            tempdirname = str(np.random.randint(1, 1000))
            try:
                os.mkdir(tempdirname)
                os.chdir(tempdirname)
                break 
            except FileExistsError:
                # Directory already exists, try again
                continue
        os.system('cp ../tde.in .')
        while energy < max_energy:
            
            # Calculate velocity and run simulation
            vel_vector = self.calculate_pka_vel(self.direction, pka_energy=energy)
            self.modify_lammps_in(filename='tde.in', pka_vector=vel_vector)
            subprocess.run(self.run_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            defect_count = self.defect_check()

            if defect_count != 0:  # defect found
                # Now step back down by 1 eV until no defect
                while defect_count != 0 and energy > 0:
                    energy -= 1
                    vel_vector = self.calculate_pka_vel(self.direction, pka_energy=energy)
                    self.modify_lammps_in(filename='tde.in', pka_vector=vel_vector)
                    subprocess.run(self.run_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    defect_count = self.defect_check()

                # Once we exit, energy is the lowest energy with a dfect found
                tde_energy = energy + 1
                break  # stop searching, we’ve bracketed the threshold

            else:
                energy += energy_increment  # keep increasing
                
        os.chdir('../')
        os.system(f'rm -rf {tempdirname}')

        return float(tde_energy)
                
    
def evaluate_tde(cfg, u: np.ndarray, start_energy = 5) -> Tuple[float, float]:

    """
    Evaluates the threshold defect energy (TDE) for a given direction `u`
    and LAMMPS configuration `cfg`.

    Parameters:
        cfg (Config): LAMMPS configuration
        u (np.ndarray): Direction of the PKA atom
        start_energy (float): Energy to start the search at (default: 5 eV)

    Returns:
        Tuple[float, float]: The TDE energy and error (0.5) for GP noise model.
    """
    energy = TDE_simulation(direction = u, run_line=cfg.run_line).run_tde_loop(start_energy = start_energy)
    return float(energy), 0.5






