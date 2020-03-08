# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:59:40 2020

@author: mu
"""

#import folium
import geopandas as gpd
import io
import json
import numpy as np
import pandas as pd
import requests
#import holoviews as hv
#hv.extension('bokeh')
#from bokeh.models import HoverTool
#from folium import FeatureGroup, LayerControl, Map, Marker
#from IPython.display import HTML,IFrame

import matplotlib
import matplotlib.cm as cm

df=pd.read_csv(r'F:\_MU\data\t0.csv')
gdf_all=gpd.read_file(r'F:\_MU\data\wh_all_point2.shp')
gdf_office=gpd.read_file(r'F:\_MU\data\wh_office_point2.shp')
gdf_edu=gpd.read_file(r'F:\_MU\data\wh_edu_point2.shp')
gdf_med=gpd.read_file(r'F:\_MU\data\wh_med_point2.shp')



#df['xyt_list'].iloc[0].remove((6, 68341))
#df['xyt_list'].iloc[0].append((6, 68341))

df_work=df[df['group']=='work']
df_school=df[df['group']=='school']
df_other=df[df['group']=='other']

'''
l=df_work['xyt_list'].shape[0]
lst=np.concatenate([df_work['xyt_list'].values.reshape(l,1), t11],axis=1)
'''
np.random.seed(1125)
import numpy.matlib

t11=1 * np.matlib.randn((df_work.shape[0], 1)) + 7
t12=0.25 * np.matlib.randn((df_school.shape[0], 1)) + 7
t13=12 * np.matlib.rand((df_other.shape[0], 1)) + 7

import matplotlib.pyplot as plt
import seaborn as sns
#plt.style.use('ggplot')
sns.set(style="darkgrid")
fig, ax = plt.subplots()

ax.hist(t11,bins=12,histtype='stepfilled',alpha=0.7,label='work')
ax.hist(t12,bins=12,histtype='barstacked',alpha=0.7,label='school')
ax.hist(t13,bins=12,histtype='barstacked',alpha=0.7,label='other')

ax.set_xlabel('h')
ax.set_ylabel('No. of Agent')
ax.set_title('Distribution of departure time')
ax.legend(prop={'size': 18})
#plt.savefig(fig, r'F:\_MU\departure_time.png')

df_work['t0']=0
df_work['p0']=df_work.home_id

df_work['t1']=df_work.apply(lambda p: (p.t1, p.home_id) ,axis=1)


df_work['t1']=t11
df_work['p1']=df_work.home_id

#t2-t3 work
n1=df_work.shape[0]
df_work['t2']=t11+ 0.75 * np.matlib.rand((df_work.shape[0], 1)) + 0.25
#df_work['p2']=[gdf_office.OBJECTID.sample() for i in range(df_work.shape[0])]
#df_work['p2']=np.zeros((n1,1))
lst2=gdf_office['OBJECTID'].sample(n1,replace=True)
df_work['p2']=lst2.values
df_work.to_csv(r'f:/_MU/df_work.csv',index=False)


df_work['t3']=df_work['t2'].values.reshape((n1,1)) + 2 * np.matlib.rand((df_work.shape[0], 1)) + 8

#t4-t5 play
df_work['t4']=df_work['t3'].values.reshape((n1,1))+0.5 * np.matlib.rand((df_work.shape[0], 1)) +0.25
df_work['t5']=(df_work['t4'].values.reshape((n1,1))+2 * np.matlib.rand((df_work.shape[0], 1)) + 0.5)%24

#t6 home
df_work['t6']=(df_work['t5'].values.reshape((n1,1))+0.5 * np.matlib.rand((df_work.shape[0], 1)) + 0.5)%24

df_work['p3']=df_work['p2']

lst4=gdf_office['OBJECTID'].sample(n1,replace=True)
df_work['p4']=lst4.values
df_work['p5']=df_work['p4']

df_work['p6']=df_work.home_id
df_work.to_csv(r'f:/_MU/df_work.csv',index=False)
"""
df['xyt_list']=df.apply(lambda p: [(0, p.home_id)],axis=1)
df_work['t2']=df_work.apply(lambda p: (p.t2, gdf_office.OBJECTID.sample()) ,axis=1)
df_work['t3']=df_work.apply(lambda p: (p.t3, gdf_office.OBJECTID.sample()) ,axis=1)
"""
########################################################
df_school['t0']=0
df_school['p0']=df_school.home_id

#df_school['t1']=df_work.apply(lambda p: (p.t1, p.home_id) ,axis=1)
df_school['t1']=t12
df_school['p1']=df_school.home_id

#t2-t3 school
n2=df_school.shape[0]
df_school['t2']=t12 + 0.5 * np.matlib.rand((df_school.shape[0], 1)) + 0.25
lst2=gdf_edu['OBJECTID'].sample(n2,replace=True)
df_school['p2']=lst2.values
df_school['t3']=df_school['t2'].values.reshape((n2,1)) + 2 * np.matlib.rand((df_school.shape[0], 1)) + 8
df_school['p3']=df_school['p2']

#t4 home
df_school['t4']=df_school['t3'].values.reshape((n2,1))+0.5 * np.matlib.rand((df_school.shape[0], 1)) +0.25
df_school['p4']=df_school.home_id
df_school.to_csv(r'f:/_MU/df_school.csv',index=False)

#######################################################
df_other['t0']=0
df_other['p0']=df_other.home_id

#df_school['t1']=df_work.apply(lambda p: (p.t1, p.home_id) ,axis=1)
df_other['t1']=t13
df_other['p1']=df_other.home_id

#t2-t3 shopping/play
n3=df_other.shape[0]
df_other['t2']=t13 + 0.5 * np.matlib.rand((df_other.shape[0], 1)) + 0.5
lst2=gdf_all['OBJECTID'].sample(n3,replace=True)
df_other['p2']=lst2.values
df_other['t3']=(df_other['t2'].values.reshape((n3,1)) + 6 * np.matlib.rand((df_other.shape[0], 1)) + 1)%24
df_other['p3']=df_other['p2']

#t4 home
df_other['t4']=(df_other['t3'].values.reshape((n3,1))+ 0.5 * np.matlib.rand((df_other.shape[0], 1)) + 0.5)%24
df_other['p4']=df_other.home_id
df_other.to_csv(r'f:/_MU/df_other.csv',index=False)

#########################################
####  stats
#########################################

df1=pd.read_csv(r'f:/_MU/df_work.csv')
df2=pd.read_csv(r'f:/_MU/df_school.csv')
df3=pd.read_csv(r'f:/_MU/df_other.csv')


#df1=df_work
df1[['t1','t2','t3','t4','t5','t6']]=df1[['t1','t2','t3','t4','t5','t6']].astype(int)

#df2=df_school
df2[['t1','t2','t3','t4']]=df2[['t1','t2','t3','t4']].astype(int)

#df3=df_other
df3[['t1','t2','t3','t4']]=df3[['t1','t2','t3','t4']].astype(int)

pv0 = df1.pivot_table(index='p0', columns='t0', values='pid', aggfunc='count')#.reset_index()
pv1 = df1.pivot_table(index='p1', columns='t1', values='pid', aggfunc='count')#.reset_index()
pv2 = df1.pivot_table(index='p2', columns='t2', values='pid', aggfunc='count')#.reset_index()
pv3 = df1.pivot_table(index='p3', columns='t3', values='pid', aggfunc='count')#.reset_index()
pv4 = df1.pivot_table(index='p4', columns='t4', values='pid', aggfunc='count')#.reset_index()
pv5 = df1.pivot_table(index='p5', columns='t5', values='pid', aggfunc='count')#.reset_index()
pv6 = df1.pivot_table(index='p6', columns='t6', values='pid', aggfunc='count')#.reset_index()

pv7 = df2.pivot_table(index='p0', columns='t0', values='pid', aggfunc='count')#.reset_index()
pv8 = df2.pivot_table(index='p1', columns='t1', values='pid', aggfunc='count')#.reset_index()
pv9 = df2.pivot_table(index='p2', columns='t2', values='pid', aggfunc='count')#.reset_index()
pv10 = df2.pivot_table(index='p3', columns='t3', values='pid', aggfunc='count')#.reset_index()
pv11 = df2.pivot_table(index='p4', columns='t4', values='pid', aggfunc='count')#.reset_index()

pv12 = df3.pivot_table(index='p0', columns='t0', values='pid', aggfunc='count')#.reset_index()
pv13 = df3.pivot_table(index='p1', columns='t1', values='pid', aggfunc='count')#.reset_index()
pv14 = df3.pivot_table(index='p2', columns='t2', values='pid', aggfunc='count')#.reset_index()
pv15 = df3.pivot_table(index='p3', columns='t3', values='pid', aggfunc='count')#.reset_index()
pv16 = df3.pivot_table(index='p4', columns='t4', values='pid', aggfunc='count')#.reset_index()

res=pd.concat([pv0,pv1,pv2,pv3,pv4,pv5,pv6,pv7,pv8,pv9,pv10,pv11,pv12,pv13,pv14,pv15,pv16], axis=1)
res.to_csv(r'f:/_MU/res.csv',index=True)

res2=res.T
res3=res2.groupby(res2.index).sum().T
res3.to_csv(r'f:/_MU/res2.csv',index=True)
"""
gdf_all.head()
res=pd.merge(gdf_all,pv0,left_on='OBJECTID',right_on='p0', how='outer')
res=pd.merge(res,pv1,left_on='OBJECTID',right_on='p1')
res=pd.merge(res,pv2,left_on='OBJECTID',right_on='p2')

res=pd.merge(res,pv3,left_on='OBJECTID',right_on='p3')
res=pd.merge(res,pv4,left_on='OBJECTID',right_on='p4')
res=pd.merge(res,pv5,left_on='OBJECTID',right_on='p5')
res=pd.merge(res,pv6,left_on='OBJECTID',right_on='p6')
res=pd.merge(res,pv7,left_on='OBJECTID',right_on='p7')
res=pd.merge(res,pv8,left_on='OBJECTID',right_on='p8')
res=pd.merge(res,pv9,left_on='OBJECTID',right_on='p9')
res=pd.merge(res,pv10,left_on='OBJECTID',right_on='p10')
res=pd.merge(res,pv11,left_on='OBJECTID',right_on='p11')
res=pd.merge(res,pv12,left_on='OBJECTID',right_on='p12')
res=pd.merge(res,pv13,left_on='OBJECTID',right_on='p13')
res=pd.merge(res,pv14,left_on='OBJECTID',right_on='p14')
res=pd.merge(res,pv15,left_on='OBJECTID',right_on='p15')
res=pd.merge(res,pv16,left_on='OBJECTID',right_on='p16')
"""
