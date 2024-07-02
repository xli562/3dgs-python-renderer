import numpy as np
from plyfile import PlyData, PlyElement

def trim_ply(input_path, output_path, max_elements=10000):
    """ Trims a .ply file to fit max_elements.
     
    Parameters:
    input_path: /.../abc.ply
    output_path:  /.../directory """

    # Load the original PLY file
    ply_data = PlyData.read(input_path)
    
    # Get the vertex element. Assuming the vertices are stored in the first element
    vertex_data = ply_data['vertex']
    
    # Determine the number of vertices to copy (minimum of 10,000 or the total number of vertices)
    num_vertices = min(max_elements, len(vertex_data))
    
    # Create a new vertex element with only the required number of vertices
    trimmed_vertices = PlyElement.describe(
        np.array(vertex_data[:num_vertices]),
        'vertex'
    )
    
    # Create a new PlyData instance with the trimmed vertex element
    trimmed_ply_data = PlyData([trimmed_vertices], text=True)
    
    # Write the new PLY file
    trimmed_ply_data.write(output_path)