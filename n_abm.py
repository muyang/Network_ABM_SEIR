# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 06:37:09 2020

@author: mu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from py3plex.core import multinet
from py3plex.core import random_generators
import matplotlib.image as mgimg
import pandas as pd
import networkx as nx
import param
import numpy.random as rnd
import geopandas as gpd
from datetime import datetime
#######################################
##       to generate test data
#######################################
gdf= pd.DataFrame(np.random.randn(150, 2) / [10, 10] + [37.76, 114.4],columns=['lat', 'lon'])
gdf['gid']=list(range(150))
gdf['virus-coming']=False
gdf['N']=(abs(np.random.randn(150, 1))*100).astype(int)


pdf=pd.DataFrame(list(range(7500)),columns=['pid'])
pdf['xy0']=gdf.gid.sample(7500,replace=True).values
pdf['xy1']=gdf.gid.sample(7500,replace=True).values
pdf['xy2']=gdf.gid.sample(7500,replace=True).values
pdf['xy3']=gdf.gid.sample(7500,replace=True).values
pdf['xy4']=gdf.gid.sample(7500,replace=True).values
pdf['xy5']=gdf.gid.sample(7500,replace=True).values
pdf['xy6']=gdf.gid.sample(7500,replace=True).values
pdf['xy7']=gdf.gid.sample(7500,replace=True).values

pdf['xy']= pdf.apply(lambda p: [p for p in pdf.iloc[p.pid,1:]],axis=1)
pdf['state']=pdf.apply(lambda p: 'I' if p.pid in [0, 995, 998, 1100] else 'S',axis=1)
pdf['color']=pdf.apply(lambda p: 'r' if p.pid in [0, 995, 998, 1100] else 'g',axis=1)

#pdf['pos']=pdf.apply(lambda p: (gdf[gdf.gid==p.xy0].lat,gdf[gdf.gid==p.xy0].lon) ,axis=1)

########################################################
## real data
########################################################
df1=pd.read_csv(r'f:/_MU/df_work.csv')
df2=pd.read_csv(r'f:/_MU/df_school.csv')
df3=pd.read_csv(r'f:/_MU/df_other.csv')

df1[['t1','t2','t3','t4','t5','t6']]=df1[['t1','t2','t3','t4','t5','t6']].astype(int)
df2[['t1','t2','t3','t4']]=df2[['t1','t2','t3','t4']].astype(int)
df3[['t1','t2','t3','t4']]=df3[['t1','t2','t3','t4']].astype(int)
'''
df1['xy']=df1.apply(lambda x: {x.t0:x.p0,
							   x.t1:x.p1,
							   x.t2:x.p2,
							   x.t3:x.p3,
							   x.t4:x.p4,
							   x.t5:x.p5,
							   x.t6:x.p6
							   },
					axis=1)
df2['xy']=df2.apply(lambda x: {x.t0:x.p0,
							   x.t1:x.p1,
							   x.t2:x.p2,
							   x.t3:x.p3,
							   x.t4:x.p4
							   },
					axis=1)
df3['xy']=df3.apply(lambda x: {x.t0:x.p0,
							   x.t1:x.p1,
							   x.t2:x.p2,
							   x.t3:x.p3,
							   x.t4:x.p4
							   },
					axis=1)
'''
keys=df1[['t0','t1','t2','t3','t4','t5','t6']].to_numpy()
values=df1[['p0','p1','p2','p3','p4','p5','p6']].to_numpy()
n=keys.shape[0]
m=keys.shape[1]
k=keys.reshape(n*m)
v=values.reshape(n*m)
arr=np.array(list(zip(k,v))).reshape((n, m, 2))
df1['xy']=pd.Series(arr.tolist())

keys=df2[['t0','t1','t2','t3','t4']].to_numpy()
values=df2[['p0','p1','p2','p3','p4']].to_numpy()
n=keys.shape[0]
m=keys.shape[1]
k=keys.reshape(n*m)
v=values.reshape(n*m)
arr=np.array(list(zip(k,v))).reshape((n, m, 2))
df2['xy']=pd.Series(arr.tolist())

keys=df3[['t0','t1','t2','t3','t4']].to_numpy()
values=df3[['p0','p1','p2','p3','p4']].to_numpy()
n=keys.shape[0]
m=keys.shape[1]
k=keys.reshape(n*m)
v=values.reshape(n*m)
arr=np.array(list(zip(k,v))).reshape((n, m, 2))
df3['xy']=pd.Series(arr.tolist())

df1.to_csv(r'f:/_MU/df_1.csv',index=False,float_format='%.6f')
df2.to_csv(r'f:/_MU/df_2.csv',index=False,float_format='%.6f')
df3.to_csv(r'f:/_MU/df_3.csv',index=False,float_format='%.6f')
cols=['pid','home_id','County','group','xy','state']
df11=df1[cols]
df22=df2[cols]
df33=df3[cols]
pdf=pd.concat([df11,df22,df33],axis=0)
pdf['pid']=range(pdf.shape[0])
pdf=pdf.reset_index()
pdf.drop('index',axis=1,inplace=True)
#pdf.to_csv(r'f:/_MU/pdf_hour.csv',index=False)
pdf.to_pickle(r'f:/_MU/pdf_pickle.pkl')
###############################################################################

gdf=gpd.read_file(r'F:\_MU\data\wh_all_point2.shp')
gdf['x']=gdf.geometry.x
gdf['y']=gdf.geometry.y
gdf=gdf[['OBJECTID','Floor','Level2','x','y']]
gdf.rename(columns={'OBJECTID':'gid'},inplace=True)
#gdf.to_csv(r'f:/_MU/gid_xy.csv',index=False)
gdf.to_pickle(r'f:/_MU/gdf_pickle.pkl')

#pdf['pos']=pdf.apply(lambda p: if ke)
###############################################################################
#pdf=pd.read_csv(r'f:/_MU/pdf_hour.csv')

'''
def func(d,h):
	lst=[abs(h-i) for i in d.keys()]
	idx=lst.index(min(lst))
	return list(p.xy.values())[idx]
'''
def func(p,h):
	t_list=[i[0] for i in p.xy]
	lst=[abs(h-t) for t in t_list]
	idx=lst.index(min(lst))
	return int(p.xy[idx][1])

for h in range(0,24):
    pdf['pos'+str(h)]=pdf.apply(lambda p: func(p,h), axis=1)
    print(h)
cols=['pos'+str(h) for h in range(24)]
pos=pdf[cols].to_numpy().tolist()
pdf['pos']=pd.Series(pos)
pdf=pdf[['pid','home_id','County','group','state','pos']]
pdf.to_pickle(r'f:/_MU/pdf_24h_pickle.pkl')

pdf=pd.read_pickle(r'f:/_MU/pdf_24h_pickle0.pkl')
pdf=pdf[['pid','home_id','County','group','state']+cols]
pdf.to_pickle(r'f:/_MU/pdf_24h.pkl')

########################################################################
#   Start from here!
########################################################################
pdf=pd.read_pickle(r'f:/_MU/pdf_24h.pkl')
gdf=pd.read_csv(r'f:/_MU/gid_xy.csv')
folder_tmp_files = r'f:/_MU/network/datasets/animation'

pdf['state'][pdf.pid == 81138] = 'I'
pdf['color'] = 'g'
pdf['color'][pdf.pid == 81138] = 'r' #pdf.apply(lambda p: 'r' if p.pid in [0, 995, 998, 1100] else 'g',axis=1)
pdf['infected_duration']=0
pdf['seviere_duration']=0

res=[]
for tick in range(0,24*31):
	h = tick % 24
	#pdf['curPos']=pdf.apply(lambda p: func(p,h), axis=1)
	#pdf['curPos']=pdf.apply(lambda p: p.pos[h], axis=1)   
	pdf['pos']=pdf['pos'+str(h)]
	place_in_risk=set()
	people_infected=pdf[(pdf.state=='E')|(pdf.state=='I')|(pdf.state=='Is')]
	for i in range(len(people_infected)):
		#place_in_risk.add(people_infected.xy.values[i][tick])
		place_in_risk.add(people_infected.pos.values[i])
	#pdf['pos']=pdf.apply(lambda p: p.xy[tick] ,axis=1)
	#gdf['virus-coming']=gdf.apply(lambda g: 'True' if g.gid in place_in_risk else 'False',axis=1)
	###d.sales[d.sales==24] = 100
	#print((tick, place_in_risk, gdf[gdf['virus-coming']==True])) #, pdf.iloc[0]['pos'])
	for pir in place_in_risk:
		agentSet=pdf[pdf.pos==pir]	#
		N = len(agentSet)  #or N=len(pdf[pdf.xy[tick]==pir])
		node_idx = agentSet.reset_index().pid.to_dict() 
		#color = agentSet.color.to_list()
		node_pid = agentSet.pid.to_dict()	#node_idx=agentSet.pid.astype(str).to_dict()
		color = agentSet.color.to_dict()
		state = agentSet.state.to_dict()
		infected_duration = agentSet.infected_duration.to_dict()
		seviere_duration = agentSet.seviere_duration.to_dict()
		#color = {str(k):v for k,v in color.items()}
		#pos = agentSet.apply(lambda p: (gdf[gdf.gid==p.xy[tick]].lat,gdf[gdf.gid==p.xy[tick].lon) ,axis=1)
		
        #pos = agentSet.apply(lambda p: (gdf[gdf.gid==pir].lat,gdf[gdf.gid==pir].lon) ,axis=1)
		
        #ER_multilayer = random_generators.random_multilayer_ER(N*10,3,0.05,directed=False)
		#fx = ER_multilayer.visualize_network(show=False)

		#fig = plt.figure()
		M=min(N,4)
		g = nx.watts_strogatz_graph(N, M, 0.2, seed=None)
		g = nx.relabel_nodes(g, node_idx)
		nx.set_node_attributes(g, node_pid, 'pid')
		nx.set_node_attributes(g, state, 'state')
		nx.set_node_attributes(g, color, 'color')
		nx.set_node_attributes(g, infected_duration, 'infected_duration')
		nx.set_node_attributes(g, seviere_duration, 'seviere_duration')
# 		fx = nx.draw_networkx(g, 
#                         node_color= list(nx.get_node_attributes(g,'color').values()), 
#                         node_size=10, 
#                         with_labels = False)
# 		plt.savefig("{}{}{}{}.png".format(folder_tmp_files,tick,pir,'before'))
		for i in g.nodes():					  
			if g.nodes[i]['state'] == 'E' or g.nodes[i]['state'] == 'I' or g.nodes[i]['state'] == 'Is':
				g.nodes[i]['infected_duration']+=1
				pdf['infected_duration'][pdf.pid==i]=g.nodes[i]['infected_duration']
				#E-->I
				#if g.nodes[i]['state'] == 'E' and g.nodes[i]['infected_duration']>=4:
				#	g.nodes[i]['state'] = 'I'
				
				for m in g.neighbors(i):
					if g.nodes[m]['state'] == 'S':
						if rnd.random() <= 0.025:	   #pInfect
							g.nodes[m]['state'] = 'E'	
							g.nodes[m]['color'] = 'orange'
                            
			if g.nodes[i]['state'] == 'E':
				if g.nodes[i]['infected_duration']>=int(rnd.normal(4,2))*24:
					g.nodes[i]['state'] = 'I'
					g.nodes[i]['color'] = 'red' 
					  
			elif g.nodes[i]['state'] == 'I':
				if rnd.random() <= 0.2:
					g.nodes[i]['state'] = 'Is'
					g.nodes[i]['color'] = 'brown'                    
					g.nodes[i]['seviere_duration']+=1
					pdf['seviere_duration'][pdf.pid==i]=g.nodes[i]['seviere_duration']
				else:	 
					if rnd.random() <= 0.02:
						g.nodes[i]['state'] == 'R'
						
			elif g.nodes[i]['state'] == 'Is':
				if g.nodes[i]['seviere_duration'] < 10*24: 
					if rnd.random() <= 0.0014:
						g.nodes[i]['state'] == 'R'
						g.nodes[i]['color'] = 'blue'                        
				else:		 
					edges = [edge for edge in g.edges() if i in edge] 
					g.nodes[i]['state'] = 'D'
					g.nodes[i]['color'] = 'grey'
					g.remove_edges_from(edges)
			
			pdf['state'][pdf.pid==i] = g.nodes[i]['state']                          
		'''		
		fig = plt.figure()
		fx = nx.draw_networkx(g, 
                        node_color= list(nx.get_node_attributes(g,'color').values()), 
                        node_size=10, 
                        with_labels = True)
		plt.savefig("{}{}{}{}.png".format(folder_tmp_files,tick,pir,'after'))
		'''
	sNum=len(pdf[pdf['state'] == 'S'])
	eNum=len(pdf[pdf['state'] == 'E'])
	iNum=len(pdf[pdf['state'] == 'I'])        
	isNum=len(pdf[pdf['state'] == 'Is'])
	rNum=len(pdf[pdf['state'] == 'R'])
	dNum=len(pdf[pdf['state'] == 'D']) 
	ts=datetime.now().strftime('%Y.%m.%d-%H:%M:%S')       
	l=(tick,sNum,eNum,iNum,isNum,rNum,dNum,ts)  
	res.append(l)
	print(l)

df_res=pd.DataFrame(res) 
df_res.columns=['tick','S','E','I','Is','R','D','ts']   
df_res.to_csv(r'f:/_MU/df_res.csv')

########################
#  using param & pyMC
########################
'''
from pymc import *
import pymc, a
training_pool = [0]
m = pymc.MCMC(a.model(training_pool))

avg_connections_people = param.Number(default=6)
isolatedRate = param.Number(default=0.1, bounds=(0, 1), )
infectRate = param.Number(default=0.3, bounds=(0, 1))
recoverRate = param.Number(default=0.05, bounds=(0, 1))
deathRate = param.Number(default=0.1, bounds=(0, 1))	
'''

'''
#amination
myImg = []
for p in imrange:
	img = mgimg.imread("{}{}.png".format(folder_tmp_files,p))
	imgPlot = plt.imshow(img)
	myimages.append([imgPlot])
myAnim = animation.ArtistAnimation(fig, myImg, interval=1000, blit=True)
myAnim.save(r"F:_MU\network\datasets\animation\animation.gif", writer='imagemagick', fps=1)
'''











'''
pdf['XYT']=np.array([
					gdf['gid'].sample(5000,replace=True),
					gdf['gid'].sample(5000,replace=True), 
					gdf['gid'].sample(5000,replace=True),
					gdf['gid'].sample(5000,replace=True),
					gdf['gid'].sample(5000,replace=True), 
					gdf['gid'].sample(5000,replace=True),
					gdf['gid'].sample(5000,replace=True),
					gdf['gid'].sample(5000,replace=True), 
					gdf['gid'].sample(5000,replace=True),
					gdf['gid'].sample(5000,replace=True),
					gdf['gid'].sample(5000,replace=True), 
					gdf['gid'].sample(5000,replace=True)					
					]).T.values
'''
