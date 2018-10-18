import bpy
import random
import math
import sys
import numpy as np
import os
import bmesh



def getAverageEdgeLength(myMesh):
    vert = myMesh.vertices

    edges_num=0
    edges_length=0
    for edge in myMesh.edges:
        edges_num+=1

        v1 = vert[edge.vertices[0]]
        v2 = vert[edge.vertices[1]]
        # x1 = v1.co[0]
        # x2 = v2.co[0]
        # y1 = v1[1]
        # y2 = v2[1]
        # z1 = v1[2]
        # z2 = v2[2]
        #el = math.sqrt(math.pow(x1-x2,2)+math.pow(v1[1]-v2[1],2)+math.pow(v1[2]-v2[2],2))
        el = math.sqrt(math.pow(v1.co[0]-v2.co[0],2)+math.pow(v1.co[1]-v2.co[1],2)+math.pow(v1.co[2]-v2.co[2],2))
        #edges_length+= edge.length
        edges_length+= el
    return edges_length/edges_num


# Run like this:

# blender --background --python generateCylinders.py -- inputfolder destfolder num_meshed std_dev

argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--"

#print(argv)
inputfolder = argv[0]
destfolder = argv[1]
num_meshes = int(argv[2])
std_dev = float(argv[3])

prim_num = 1    # Number of blender primitive added to existing meshes in the input folder

#num_meshes = 1
#std_dev = 0.02
#destfolder = "/morpheo-nas/marmando/DeepMeshRefinement/BlenderDB/"

# Get scene
scene = bpy.context.scene

# get list of all files in directory
file_list = sorted(os.listdir(inputfolder))

# get a list of files ending in 'obj'
obj_list = [item for item in file_list if item.endswith('.obj')]

for m in range(num_meshes):

    # # loop through the strings in obj_list and add the files to the scene
    # for item in obj_list:
    #     path_to_file = os.path.join(inputfolder, item)
    #     bpy.ops.import_scene.obj(filepath = path_to_file)


    # generate random parameters

    # Primitive type (for blender primitive) or mesh index (for meshes in input folder)
    mode = random.randint(0,prim_num+len(obj_list)-1)
    # Noise type (normal, uniform, gamma)
    noise_type = random.randint(0, 2)
    #primitive parameters (radius, height) for blender primitives (not super usefull now)
    rad = random.random()
    rad2 = random.random()*rad        
    length= random.random()
    #rotation angles
    alpha = 2*math.pi*random.random()
    beta = 2*math.pi*random.random()
    gamma = 2*math.pi*random.random()
    #test: randomize noise level
    rand_std_dev = random.random()*std_dev

    
    for obj in bpy.data.objects:
        obj.tag = False

    # if mode==0:
    #     # Create monkey
    #     bpy.ops.mesh.primitive_monkey_add(radius = rad)
    #     gt_name = 'monkey_'+str(m)+'_gt'
    #     noisy_name = 'monkey_'+str(m)+'_noisy'
    if mode==0:
        # Create torus
        bpy.ops.mesh.primitive_torus_add(major_radius = rad, minor_radius= rad2)
        gt_name = 'torus_'+str(m)+'_gt'
        noisy_name = 'torus_'+str(m)+'_noisy'
    # elif mode==2:
    #     # Create sphere
    #     bpy.ops.mesh.primitive_uv_sphere_add(size = rad)
    #     gt_name = 'sphere_'+str(m)+'_gt'
    #     noisy_name = 'sphere_'+str(m)+'_noisy'
    else:
        # Import obj from input dir
        item = obj_list[mode-prim_num]
        path_to_file = os.path.join(inputfolder, item)
        bpy.ops.import_scene.obj(filepath = path_to_file)
        gt_name = item[:-4]+'_'+str(m)+'_gt'
        noisy_name = item[:-4]+'_'+str(m)+'_noisy'
    
    # if mode==0:
    #     # Create cylinder
    #     bpy.ops.mesh.primitive_cylinder_add(radius = rad, depth = length)
    #     gt_name = 'cylinder_'+str(m)+'_gt'
    #     noisy_name = 'cylinder_'+str(m)+'_noisy'
    # elif mode==1:
    #     # Create cone
    #     bpy.ops.mesh.primitive_cone_add(radius1 = rad, radius2= rad2, depth = length)
    #     gt_name = 'cone_'+str(m)+'_gt'
    #     noisy_name = 'cone_'+str(m)+'_noisy'
    # elif mode==2:
    #     # Create cube
    #     bpy.ops.mesh.primitive_cube_add(radius = rad)
    #     gt_name = 'cube_'+str(m)+'_gt'
    #     noisy_name = 'cube_'+str(m)+'_noisy'
    # elif mode==3:
    #     # Create icosahedron
    #     bpy.ops.mesh.primitive_ico_sphere_add(size=rad, subdivisions=1)
    #     gt_name = 'ico_'+str(m)+'_gt'
    #     noisy_name = 'ico_'+str(m)+'_noisy'
    
    if mode>(prim_num-1):
        imported_objects = [obj for obj in bpy.data.objects if ((obj.tag is True)and(obj.type=='MESH'))]
        ob = imported_objects[0]
    else:
        # Added primitive will be the active object afterits created
        ob = bpy.context.object
    
    me = ob.data
    ob.name = gt_name
    
    #print("m = "+str(m)+", mode = "+str(mode))
    print("ob name = "+str(ob.name))

    # These two lines are unnecessary and will generate another copy of your cylinder
    #ob = bpy.data.objects.new(Name, me1)
    #scj.objects.link(ob)

    # If you want to move your object, simply set its location thus:
#    d_x = random.random()
#    d_y = random.random()
#    d_z = random.random()
#    ob.location = ( d_x, d_y, d_z )

    # Rotate cylinder
    ob.rotation_euler = (alpha,beta,gamma)
    
    # Copy mesh
    me = ob.data
    me_copy = me.copy()
    ob_noise = bpy.data.objects.new(noisy_name, me_copy)
    
    
    scene.objects.link(ob_noise)
    scene.update()
    
    # Move and rotate copy, to match GT
    ob_noise.rotation_euler = (alpha,beta,gamma)
    ob_noise.location = ob.location

    mesh = ob_noise.data
    
    # --- Add noise ---
    
    el = getAverageEdgeLength(mesh)
    # Select noise type randomly
    if noise_type==0:   # Normal
        disp = rand_std_dev*el * np.random.normal(0,1,(len(mesh.vertices),3))
    elif noise_type==1: # Uniform
        disp = rand_std_dev*el * np.random.uniform(-1,1,(len(mesh.vertices),3))
    elif noise_type==2: # Gamma
        disp = rand_std_dev*el * np.random.gamma(2.0,2.0,(len(mesh.vertices),3))



    for vert in range(len(mesh.vertices)):
        for coord in range(3):
            #this_disp = disp[vert,coord]
            mesh.vertices[vert].co[coord]+=disp[vert,coord]
    
    #print(bpy.data)

    print("Noise type: "+str(noise_type))

    # deselect all objects
    bpy.ops.object.select_all(action='DESELECT')

    # make the current object active and select it
    scene.objects.active = ob
    ob.select = True
    # export the currently selected object to its own file based on its name
    bpy.ops.export_scene.obj(filepath = destfolder + ob.name + ".obj", use_selection = True, use_edges=False, use_normals=False, use_uvs=False, use_materials=False, keep_vertex_order=True)

    # remove ob
    bpy.ops.object.delete() 

    # make the current object active and select it
    scene.objects.active = ob_noise
    ob_noise.select = True
    # export the currently selected object to its own file based on its name
    bpy.ops.export_scene.obj(filepath = destfolder + ob_noise.name + ".obj", use_selection = True, use_edges=False, use_normals=False, use_uvs=False, use_materials=False, keep_vertex_order=True)

    # remove ob
    bpy.ops.object.delete() 


# # --- Exporting part ---
# # deselect all objects
# bpy.ops.object.select_all(action='DESELECT')    


# # loop through all the objects in the scene
# scene = bpy.context.scene
# for ob in scene.objects:
#     # make the current object active and select it
#     scene.objects.active = ob
#     ob.select = True
    
#     # make sure that we only export meshes
#     if ob.type == 'MESH':
#         # export the currently selected object to its own file based on its name
#         bpy.ops.export_scene.obj(filepath = destfolder + ob.name + ".obj", use_selection = True, use_edges=False, use_normals=False, use_uvs=False, use_materials=False, keep_vertex_order=True)
#     # deselect the object and move on to another if any more are left
#     ob.select = False