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
#import geopandas as gpd
from datetime import datetime

###########################################
pdf=pd.read_pickle(r'f:/_MU/pdf_24h.pkl')
gdf=pd.read_csv(r'f:/_MU/gid_xy.csv')
folder_tmp_files = r'F:\_MU\network\datasets\animation'

pdf['state'][pdf.pid == 81138] = 'I'
pdf['color'] = 'g'
pdf['color'][pdf.pid == 81138] = 'r' #pdf.apply(lambda p: 'r' if p.pid in [0, 995, 998, 1100] else 'g',axis=1)
pdf['infected_duration']=0
pdf['seviere_duration']=0
pdf['hospital_duration']=0
res=[]
R0=[]

for tick in range(0,24*31,3):
	h = tick % 24
	#pdf['curPos']=pdf.apply(lambda p: func(p,h), axis=1)
	#pdf['curPos']=pdf.apply(lambda p: p.pos[h], axis=1)   
	pdf['pos']=pdf['pos'+str(h)]
	place_in_risk=set()
	people_infected=pdf.loc[(pdf.state=='E')|(pdf.state=='I')|(pdf.state=='Is')]
	#pdf.loc[pdf.state=='Is']['pos']=pdf.loc[pdf.state=='Is']['home_id']
	for i in range(len(people_infected)):
		#place_in_risk.add(people_infected.xy.values[i][tick])
		place_in_risk.add(people_infected.pos.values[i])
	#pdf['pos']=pdf.apply(lambda p: p.xy[tick] ,axis=1)
	#gdf['virus-coming']=gdf.apply(lambda g: 'True' if g.gid in place_in_risk else 'False',axis=1)
	###d.sales[d.sales==24] = 100
	#print((tick, place_in_risk, gdf[gdf['virus-coming']==True])) #, pdf.iloc[0]['pos'])
	for pir in place_in_risk:
		agentSet=pdf.loc[pdf.pos==pir][['pid','group','state','color','infected_duration','seviere_duration','hospital_duration']]	#
		N = len(agentSet)  #or N=len(pdf[pdf.xy[tick]==pir])
		node_idx = agentSet.reset_index().pid.to_dict() 
		node_pid = agentSet.pid.to_dict()	#node_idx=agentSet.pid.astype(str).to_dict()
		color = agentSet.color.to_dict()
		state = agentSet.state.to_dict()
		infected_duration = agentSet.infected_duration.to_dict()
		seviere_duration = agentSet.seviere_duration.to_dict()
		hospital_duration = agentSet.hospital_duration.to_dict()        
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
		nx.set_node_attributes(g, hospital_duration, 'hospital_duration')
# 		fx = nx.draw_networkx(g, 
#                         node_color= list(nx.get_node_attributes(g,'color').values()), 
#                         node_size=10, 
#                         with_labels = False)
# 		plt.savefig("{}{}{}{}.png".format(folder_tmp_files,tick,pir,'before'))
		r0_i=[]
		for i in g.nodes():
			g.nodes[i]['R0'] = 0				  
			if g.nodes[i]['state'] == 'E' or g.nodes[i]['state'] == 'I' or g.nodes[i]['state'] == 'Is':
				g.nodes[i]['infected_duration']+=1*3
				#pdf['infected_duration'][pdf.pid==i]=g.nodes[i]['infected_duration']
				pdf.loc[pdf.pid==i,'infected_duration']=g.nodes[i]['infected_duration']                
				#E-->I
				#if g.nodes[i]['state'] == 'E' and g.nodes[i]['infected_duration']>=4:
				#	g.nodes[i]['state'] = 'I'
				
				for j in g.neighbors(i):
					if g.nodes[j]['state'] == 'S':
						if rnd.random() <= 0.025:	   #pInfect
							g.nodes[j]['state'] = 'E'	
							g.nodes[j]['color'] = 'orange'
							pdf['state'][pdf.pid==j] ='E'
							g.nodes[i]['R0'] += 1
				r0_i.append(g.nodes[i]['R0'])
                           
			if g.nodes[i]['state'] == 'E':
				if g.nodes[i]['infected_duration']>=int(rnd.normal(4,2))*24/3:  #1tick=3h
					g.nodes[i]['state'] = 'I'
					g.nodes[i]['color'] = 'red' 
					pdf['state'][pdf.pid==i]='I'
					  
			elif g.nodes[i]['state'] == 'I':
				if rnd.random() <= 0.013 * 3 :    #0.013
					g.nodes[i]['state'] = 'Is'
					g.nodes[i]['color'] = 'brown' 
					pdf['state'][pdf.pid==i]='Is'                    
					g.nodes[i]['seviere_duration']+=1*3
					#pdf['seviere_duration'][pdf.pid==i]=g.nodes[i]['seviere_duration']
					pdf.loc[pdf.pid==i,'seviere_duration']=g.nodes[i]['seviere_duration']                    
				else:	 
					if rnd.random() <= 0.003 * 3:
						g.nodes[i]['state'] == 'R'
						pdf['state'][pdf.pid==i] ='R'                         
						
			elif g.nodes[i]['state'] == 'Is':
				if  rnd.random() < 0.06:  
					g.nodes[i]['state'] = 'H'
					g.nodes[i]['color'] = 'yellow'
					pdf['state'][pdf.pid==i]='H'                    
				elif g.nodes[i]['seviere_duration'] < 7*24/3:                #2 in 1
					if rnd.random() <= 0.0014 * 3:
						g.nodes[i]['state'] == 'R'
						g.nodes[i]['color'] = 'blue'
						pdf['state'][pdf.pid==i] ='R'                         
				else:		 
					edges = [edge for edge in g.edges() if i in edge] 
					g.nodes[i]['state'] = 'D'
					g.nodes[i]['color'] = 'grey'
					pdf['state'][pdf.pid==i]='D'                    
					g.remove_edges_from(edges)

			elif g.nodes[i]['state'] == 'H': 
				g.nodes[i]['hospital_duration']+=1*3                
				if g.nodes[i]['hospital_duration']<15*24/3:   #15 day in hospital
					if rnd.random() <= 0.003 * 3:
						g.nodes[i]['state'] == 'R'
						g.nodes[i]['color'] = 'blue'   
						pdf['state'][pdf.pid==i] ='R'                         
				else:		 
					edges = [edge for edge in g.edges() if i in edge] 
					g.nodes[i]['state'] = 'D'
					g.nodes[i]['color'] = 'grey'
					pdf['state'][pdf.pid==i]='D'                    
					g.remove_edges_from(edges)
                
			#pdf['state'][pdf.pid==i] = g.nodes[i]['state']
			#pdf.loc[pdf.pid==i,'state'] = g.nodes[i]['state']                           
		'''		  
		fig = plt.figure()
		fx = nx.draw_networkx(g, 
                        node_color= list(nx.get_node_attributes(g,'color').values()), 
                        node_size=10, 
                        with_labels = True)
		plt.savefig("{}{}{}{}.png".format(folder_tmp_files,tick,pir,'after'))
		'''
	R0.append(np.mean(r0_i))     #R0.append(np.mean(r0_i) if r0_i else 0)      
	sNum=len(pdf.loc[pdf.state == 'S'])
	eNum=len(pdf.loc[pdf.state == 'E'])
	iNum=len(pdf.loc[pdf.state == 'I'])        
	isNum=len(pdf.loc[pdf.state == 'Is'])
	hNum=len(pdf.loc[pdf.state == 'H'])    
	rNum=len(pdf.loc[pdf.state == 'R'])
	dNum=len(pdf.loc[pdf.state == 'D']) 
	ts=datetime.now().strftime('%Y.%m.%d-%H:%M:%S')       
	l=(tick,sNum,eNum,iNum,isNum,hNum,rNum,dNum,np.mean(R0))  
	res.append(l)
	print(l,ts)

df_res=pd.DataFrame(res) 
df_res.columns=['tick','S','E','I','Is','H','R','D','infected','place_in_risk']   
