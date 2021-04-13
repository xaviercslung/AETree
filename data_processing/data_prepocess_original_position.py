#!/usr/bin/env python
# coding: utf-8

# In[2]:


import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
get_ipython().magic(u'matplotlib inline')
# %matplotlib notebook


# In[3]:


def draw_ploygon(xyz_foot):
    x = xyz_foot[:,0]
    y = xyz_foot[:,1]
    z = xyz_foot[:,2]

    ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
    ax.axis('off')
    ax.scatter(x, y, z, c='g',alpha=1)  # 绘制数据点,颜色是红色
    ax.plot(x, y, z, c='b')
    ax.plot([x[-1],x[0]], [y[-1],y[0]], [z[-1],z[0]], c='r')

    ax.set_xlabel('X')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
#     plt.show()


# In[4]:


filelist = sorted(glob.glob("../Data/building_polygon/*"))


# In[5]:


len(filelist)


# In[8]:


# 1233
filelist = sorted(glob.glob("../Data/building_polygon/*"))
# 45847
# filelist = sorted(glob.glob("../../../data/RealCity3D/final_dataset/Finished_Poly/New_York/Manhattan/*"))


# In[9]:


foot_xyz_list = []

for index_file,file in enumerate(filelist):
    with open(file, 'r') as f:
        print(index_file, file)
        lines = f.readlines()

        # get center xy
        center_x, center_y = lines[2].replace('\n','').split(' ')[2:]
        center_x = float(center_x)
        center_y = float(center_y)

        # get footprint line
        for i in range(len(lines)):
            if((lines[i].find('f'))!=-1):
#                 print(lines[i])
                break
        face_index = np.array([float(v) for v in lines[i].replace('\n','').split(' ')[1:]]).astype('int') - 1    

        # put xyz into array
        n = len(lines[4:i])
        xyz = np.zeros((n,3))
        for j in range(n):
            x, y, z = lines[4+j].replace('\n','').split(' ')[1:]
            xyz[j,:] = float(x), float(y), float(z)

        # get footprint xyz
        xyz_foot = xyz[face_index]
        xyz_foot[:,0] = xyz_foot[:,0] + center_x
        xyz_foot[:,1] = xyz_foot[:,1] + center_y
        xyz_foot[:,2] = 0  
        foot_xyz_list.append(xyz_foot)
        #vertify it's planar or not
#         print(len(np.unique(xyz_foot[:,2])))
#         if(len(np.unique(xyz_foot[:,2]))==1):
#             good_n += 1
#         else:
#             bad_n += 1


# In[10]:


# ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程

fig = plt.figure(figsize=(20,15))
ax = fig.gca(projection='3d')
ax.axis('off')
# fig = plt.figure()
# ax = Axes3D(fig)
for i, xyz_foot in enumerate(foot_xyz_list):
    xyz_foot = foot_xyz_list[i]
    x = xyz_foot[:,0]
    y = xyz_foot[:,1]
    z = xyz_foot[:,2]
#     print(x,y,z)
  
    ax.scatter(x, y, z, c='g',alpha=1)  # 绘制数据点,颜色是红色
    ax.plot(x, y, z, c='b')
    ax.plot([x[-1],x[0]], [y[-1],y[0]], [z[-1],z[0]], c='r')
#     plt.show()
# xyz_foot = foot_xyz_list[1]
# x = xyz_foot[:,0]
# y = xyz_foot[:,1]
# z = xyz_foot[:,2]

# ax.axis('off')
# ax.scatter(x, y, z, c='g',alpha=1)  # 绘制数据点,颜色是红色
# ax.plot(x, y, z, c='b')
# ax.plot([x[-1],x[0]], [y[-1],y[0]], [z[-1],z[0]], c='r')
#     ax.set_xlabel('X')  # 坐标轴
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')    
plt.savefig('test.png')
plt.show()


# In[36]:


norm_foot_xyz_list = []
for xyz_foot in foot_xyz_list:
    xyz_foot = (xyz_foot - np.min(xyz_foot, axis=0))/(np.max(xyz_foot, axis=0) - np.min(xyz_foot, axis=0))
    xyz_foot[:,2] = 0
    norm_foot_xyz_list.append(xyz_foot)


# In[28]:


filename = 'Manhattan_polygon2_1233.pickle'
pfile = open(filename,'wb')
pickle.dump(foot_xyz_list, pfile)
pfile.close()


# In[50]:


filename = 'Manhattan_polygon2_1233_pickle2.pickle'
pfile = open(filename,'wb')
pickle.dump(foot_xyz_list, pfile, protocol=2)
pfile.close()


# In[39]:


filename = 'Manhattan_polygon2_1233_norm.pickle'
pfile = open(filename,'wb')
pickle.dump(norm_foot_xyz_list, pfile)
pfile.close()


# In[ ]:





# In[58]:


# 1233
# filelist = sorted(glob.glob("../Data/building_polygon/*"))
# 45847
filelist_all = sorted(glob.glob("../../../data/RealCity3D/final_dataset/Finished_Poly/New_York/Manhattan/*"))


# In[59]:


foot_xyz_list_all = []

for index_file,file in enumerate(filelist_all):
    with open(file, 'r') as f:
        print(index_file, file)
        lines = f.readlines()

        # get center xy
        center_x, center_y = lines[2].replace('\n','').split(' ')[2:]
        center_x = float(center_x)
        center_y = float(center_y)

        # get footprint line
        for i in range(len(lines)):
            if((lines[i].find('f'))!=-1):
#                 print(lines[i])
                break
        face_index = np.array([float(v) for v in lines[i].replace('\n','').split(' ')[1:]]).astype('int') - 1    

        # put xyz into array
        n = len(lines[4:i])
        xyz = np.zeros((n,3))
        for j in range(n):
            x, y, z = lines[4+j].replace('\n','').split(' ')[1:]
            xyz[j,:] = float(x), float(y), float(z)

        # get footprint xyz
        xyz_foot = xyz[face_index]
        xyz_foot[:,0] = xyz_foot[:,0] + center_x
        xyz_foot[:,1] = xyz_foot[:,1] + center_y
        xyz_foot[:,2] = 0  
        foot_xyz_list_all.append(xyz_foot)


# In[63]:


len(foot_xyz_list_all)


# In[64]:


norm_foot_xyz_list_all = []
for xyz_foot in foot_xyz_list_all:
    xyz_foot = (xyz_foot - np.min(xyz_foot, axis=0))/(np.max(xyz_foot, axis=0) - np.min(xyz_foot, axis=0))
    xyz_foot[:,2] = 0
    if(np.isnan(np.min(xyz_foot))):
        print((np.max(xyz_foot, axis=0) - np.min(xyz_foot, axis=0)), xyz_foot)
    norm_foot_xyz_list_all.append(xyz_foot)


# In[46]:


filename = 'Manhattan_polygon2_45847.pickle'
pfile = open(filename,'wb')
pickle.dump(foot_xyz_list_all, pfile)
pfile.close()


# In[61]:


filename = 'Data/Manhattan_polygon2_45847_norm.pickle'
pfile = open(filename,'wb')
pickle.dump(norm_foot_xyz_list_all, pfile)
pfile.close()


# In[65]:


for i in range(len(norm_foot_xyz_list_all)):
    if(np.isnan(np.sum(norm_foot_xyz_list_all[i]))):
        print(i, norm_foot_xyz_list_all[i])


# In[52]:


ls ../../../data/RealCity3D/final_dataset/Finished_Poly/Zurich/


# In[55]:


Z_filelist_all = sorted(glob.glob("../../../data/RealCity3D/final_dataset/Finished_Poly/Zurich/*"))


# In[57]:


Z_foot_xyz_list_all = []

for index_file,file in enumerate(Z_filelist_all):
    with open(file, 'r') as f:
        print(index_file, file)
        lines = f.readlines()

        # get center xy
        center_x, center_y = lines[2].replace('\n','').split(' ')[2:]
        center_x = float(center_x)
        center_y = float(center_y)

        # get footprint line
        for i in range(len(lines)):
            if((lines[i].find('f'))!=-1):
#                 print(lines[i])
                break
        face_index = np.array([float(v) for v in lines[i].replace('\n','').split(' ')[1:]]).astype('int') - 1    

        # put xyz into array
        n = len(lines[4:i])
        xyz = np.zeros((n,3))
        for j in range(n):
            x, y, z = lines[4+j].replace('\n','').split(' ')[1:]
            xyz[j,:] = float(x), float(y), float(z)

        # get footprint xyz
        xyz_foot = xyz[face_index]
        xyz_foot[:,0] = xyz_foot[:,0] + center_x
        xyz_foot[:,1] = xyz_foot[:,1] + center_y
        xyz_foot[:,2] = 0  
        Z_foot_xyz_list_all.append(xyz_foot)


# In[ ]:




