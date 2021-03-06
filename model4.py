# -*- coding: utf-8 -*-
"""
Created on Fri Mar	6 06:37:09 2020

@author: mu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from py3plex.core import multinet
#from py3plex.core import random_generators
import matplotlib.image as mgimg
import pandas as pd
#import modin.pandas as pd
import networkx as nx
import param
import numpy.random as rnd
#import geopandas as gpd
from datetime import datetime
from numba import cuda
import h5py
###########################################
	
'''
['pid', 'home_id', 'County', 'group', 'state', 'pos0', 'pos1', 'pos2', 'pos3', 'pos4',
 'pos5', 'pos6', 'pos7', 'pos8', 'pos9', 'pos10', 'pos11', 'pos12', 'pos13', 'pos14', 
 'pos15', 'pos16', 'pos17', 'pos18', 'pos19', 'pos20', 'pos21', 'pos22', 'pos23', 'infected_duration',
 'seviere_duration', 'hospital_duration', 'onset_duration', 'native','leave', 'Hospitalization', 'pos']
'''
'''
gid,Floor,Level2,x,y
'''
#@cuda.jit
def step(ticks,interval,param,pdf,gdf):
	for tick in range(0,ticks,interval):
		if tick > 24*40:
			pdf = pdf.loc[pdf.leave >= int(tick/24) - 40]
			
		h = tick % 24
		pdf.loc[(pdf.state!='Is')&(pdf.Hospitalization==False),'pos']=pdf['pos'+str(h)]
		place_in_risk=set()
		if h < 8 and h > 22:
			pdf.loc[(pdf.state=='Is')&(pdf.Hospitalization==False),'pos']=pdf['home_id']
		else:
			l=pdf.loc[(pdf.state=='Is')&(pdf.Hospitalization==False),'pos'].shape[0]
			pdf.loc[(pdf.state=='Is')&(pdf.Hospitalization==False),'pos']=gdf.gid.sample(l,replace=True)
		
		people_infected=pdf.loc[(pdf.state.isin(['E','I','Is']))&(pdf.Hospitalization==False)]
		x1=people_infected.shape[0]
		x2=0

		for i in range(x1):
			place_in_risk.add(people_infected.pos.values[i])
		for pir in place_in_risk:
			agentSet=pdf.loc[pdf.pos==pir][['pid','group','state','infected_duration','seviere_duration','native','Hospitalization','pos']]	#
			#print(gdf[gdf.gid==pir].Level2)
			placeType=gdf.loc[gdf.gid==pir,'Level2'].values
			floorNum=gdf.loc[gdf.gid==pir,'Floor'].values
			print(placeType,floorNum)
			N = agentSet.shape[0] #or N=len(pdf[pdf.xy[tick]==pir])

			node_idx = agentSet.reset_index().pid.to_dict() 
			node_pid = agentSet.pid.to_dict()	#node_idx=agentSet.pid.astype(str).to_dict()
			#color = agentSet.color.to_dict()
			state = agentSet.state.to_dict()
			pos = agentSet.pos.to_dict()
			infected_duration = agentSet.infected_duration.to_dict()
			seviere_duration = agentSet.seviere_duration.to_dict()
			#hospital_duration = agentSet.hospital_duration.to_dict()
			Hospitalization =  agentSet.Hospitalization.to_dict()
			#ICU =  agentSet.ICU.to_dict()		 
			#pos = agentSet.apply(lambda p: (gdf[gdf.gid==p.xy[tick]].lat,gdf[gdf.gid==p.xy[tick].lon) ,axis=1)

			#ER_multilayer = random_generators.random_multilayer_ER(N*10,3,0.05,directed=False)
			#fx = ER_multilayer.visualize_network(show=False)
	
			#fig = plt.figure()
			g=nx.empty_graph()
			#g.add_nodes_from(agentSet[:,0])
			if N>1:
				if placeType==101:
					if h > 5 and h < 23:
						M=min(N,param[0])    #m1=param[0]
						g = nx.watts_strogatz_graph(N, M, param[3], seed=None)
				elif placeType==402:
					M=min(N,param[1])
					g = nx.watts_strogatz_graph(N, M, param[4], seed=None)
				else:
					M=min(N,param[2])
					g = nx.watts_strogatz_graph(N, M, param[5], seed=None)
					#g = random_generators.random_multilayer_ER(N,floorNum,0.1,directed=False)
			
			g = nx.relabel_nodes(g, node_idx)
			nx.set_node_attributes(g, node_pid, 'pid')
			nx.set_node_attributes(g, state, 'state')
			#nx.set_node_attributes(g, color, 'color')
			nx.set_node_attributes(g, infected_duration, 'infected_duration')
			nx.set_node_attributes(g, seviere_duration, 'seviere_duration')
			nx.set_node_attributes(g, pos, 'pos')
			nx.set_node_attributes(g, Hospitalization, 'Hospitalization')
			'''
			g = nx.relabel_nodes(g, dict(zip(range(N),list(agentSet[:,0]))))
			nx.set_node_attributes(g, dict(agentSet[:,[0,1]]), 'pid')
			nx.set_node_attributes(g, dict(agentSet[:,[0,4]]), 'state')
			nx.set_node_attributes(g, dict(agentSet[:,[0,29]]), 'infected_duration')
			nx.set_node_attributes(g, dict(agentSet[:,[0,30]]), 'seviere_duration')
			#nx.set_node_attributes(g, dict(agentSet[:,[0,31]]), 'hospital_duration')
			nx.set_node_attributes(g, dict(agentSet[:,[0,35]]), 'Hospitalization')
			'''
			#nx.set_node_attributes(g, ICU, 'ICU')
	#		fx = nx.draw_networkx(g, 
	#						  node_color= list(nx.get_node_attributes(g,'color').values()), 
	#						  node_size=10, 
	#						  with_labels = False)
	#		plt.savefig("{}{}{}{}.png".format(folder_tmp_files,tick,pir,'before'))
			#r0_i=[]
			#print('t2:',datetime.now().strftime('%Y.%m.%d-%H:%M:%S'))	
			for i in g.nodes():
				#g.nodes[i]['R0'] = 0				  
				if g.nodes[i]['state'] == 'E' or g.nodes[i]['state'] == 'I' or g.nodes[i]['state'] == 'Is':
					g.nodes[i]['infected_duration']+=1*interval
					#pdf['infected_duration'][pdf.pid==i]=g.nodes[i]['infected_duration']
					pdf.loc[pdf.pid==i,'infected_duration']=g.nodes[i]['infected_duration']		
	
					for j in g.neighbors(i):
						if g.nodes[j]['state'] == 'S':
							if rnd.random() <= param[6]:	   #param[6]=pInfect
								pdf.loc[pdf.pid==j,'state'] ='E'
								x2 += 1
							
								#g.nodes[i]['R0'] += 1
					#r0_i.append(g.nodes[i]['R0'])
				
				if  g.nodes[i]['state'] == 'E': #data[i,4]=='E':
					if g.nodes[i]['infected_duration'] >=4*24/interval:     #S-E 4days
						pdf.loc[pdf.pid==i,'state'] ='I'
						
				if g.nodes[i]['state'] == 'I': #data[i,4]=='I':
					if rnd.random() <= param[7]/24/param[8] * interval:	      #param[7][8] 88%,8 days，
						g.nodes[i]['Hospitalization']=True
						pdf.loc[pdf.pid==i,'Hospitalization']=True
						pdf.loc[pdf.pid==i,'pos']=gdf.gid.sample(1)
						
					if g.nodes[i]['Hospitalization']==True:					
						if g.nodes[i]['infected_duration'] > (rnd.uniform(6,10))*24/interval:						 
							pdf.loc[pdf.pid==i,'state'] ='Is'
						elif rnd.normal() <= 0.81/24/(rnd.uniform(6,10)) * interval:
							pdf.loc[pdf.pid==i,'state'] ='R'
					else:
						if g.nodes[i]['infected_duration'] > (rnd.uniform(2,9))*24/interval:
							pdf.loc[pdf.pid==i,'state'] ='Is'
						elif rnd.normal() <=  0.81/24/(rnd.uniform(2,9)) * interval:         # community 2-9 days
							pdf.loc[pdf.pid==i,'state'] ='R'
					
				if g.nodes[i]['state'] =='Is': #data[i,4]=='Is':
					g.nodes[i]['seviere_duration']+=1*interval
					if rnd.random() <= 1/24 * interval:	      #1 days，100%
						g.nodes[i]['Hospitalization']=True
						pdf.loc[pdf.pid==i,'Hospitalization']=True
						pdf.loc[pdf.pid==i,'pos']=gdf.gid.sample(1)						
					if g.nodes[i]['Hospitalization']==True:
						if g.nodes[i]['seviere_duration'] > int(rnd.uniform(13,20))*24/interval:			 #2 in 1	 
							pdf.loc[pdf.pid==i,'state'] ='D'
							
						elif rnd.random() <	param[9]/24/(rnd.uniform(13,20)) * interval:	#  after 13-20 day in H 
							pdf.loc[pdf.pid==i,'state'] ='R'  
							 
					elif g.nodes[i]['seviere_duration'] > (rnd.uniform(6,10))*24/interval:			 #2 in 1	 
						pdf.loc[pdf.pid==i,'state'] ='D'
						
			'''
			fig = plt.figure()
			fx = nx.draw_networkx(g, 
							node_color= list(nx.get_node_attributes(g,'color').values()), 
							node_size=10, 
							with_labels = True)
			plt.savefig("{}{}{}{}.png".format(folder_tmp_files,tick,pir,'after'))
			'''
		#R0.append(np.mean(r0_i))	  #R0.append(np.mean(r0_i) if r0_i else 0)		
		sNum=pdf.loc[pdf.state == 'S'].shape[0]
		eNum=pdf.loc[pdf.state == 'E'].shape[0]
		iNum=pdf.loc[pdf.state == 'I'].shape[0]
		isNum=pdf.loc[pdf.state == 'Is'].shape[0]
		#hNum=len(pdf.loc[pdf.state == 'H'])	   
		rNum=pdf.loc[pdf.state == 'R'].shape[0]
		dNum=pdf.loc[pdf.state == 'D'].shape[0]
		iaNum1=pdf.loc[(pdf.state != 'S')&(pdf.native==True)].shape[0]
		iaNum2=pdf.loc[(pdf.state != 'S')&(pdf.native==False)].shape[0]		  
		#R_0 = beta*N / gamma = N*ln(S(0)/S(t)) / (K-S(t))
		#R_0=np.log((7352961/sNum))/(7352962-sNum) * 7352961	
		#x2=eNum+iNum+isNum 
		if x1>0:
			R0=x2/x1
		else:
			R0=0
		ts=datetime.now().strftime('%Y.%m.%d-%H:%M:%S')		  
		l=(tick,sNum,eNum,iNum,isNum,rNum,dNum,iaNum1,iaNum2,R0)	
		res.append(l)
		print(l,ts)	  
		 
	df_res=pd.DataFrame(res) 
	df_res.columns=['tick','S','E','I','Is','R','D','native','non-native','R0']	  
	df_res.to_csv(r'f:/_MU/res.csv')
	with h5py.File(r'f:/_MU/data.h5','w') as hf:
		hf.create_dataset('data',data=pdf.values)

	
def main():
	#global X, width, height, P, S_ij, args, a, b, radius, size, directory, Dmn_copy, nb_type, nb_type_index, id #,Dmn
	#global X16_obs, X16_sim, X86
	#params=sys.argv
	np.random.seed(1125)
	
	#init()
	pdf=pd.read_pickle(r'f:/_MU/pdf_24h.pkl')
	gdf=pd.read_csv(r'f:/_MU/gid_xy.csv')
	gdf.loc[gdf.Level2.isna(),'Level2']=101
	
	folder_tmp_files = r'f:\_MU\network\datasets\animation'
	
	#pdf.loc[pdf.pid == 81138,'state'] = 'I'
	
	E_list=pdf.pid.sample(10)
	pdf.loc[pdf.pid.isin(E_list),'state'] = 'E'
	#pdf.apply(lambda p: 'r' if p.pid in [0, 995, 998, 1100] else 'g',axis=1)
	pdf['infected_duration']=0
	#pdf['onset_duration']=8   #before 2-4 8 days,after 2 days
	pdf['seviere_duration']=0
	pdf['hospital_duration']=0
	pdf['onset_duration']=0
	pdf['native']=pdf.apply(lambda p: False if rnd.random() <= 0.357 else True, axis=1)
	l=len(pdf[pdf.native==False])
	pdf['leave']=9999
	pdf['leave'][pdf.native==False] = (rnd.random(l)*15).astype(int)
	#pdf['family_id']
	pdf['Hospitalization']=False
	#pdf['ICU']=False
	pdf['pos']=pdf.home_id	
	#d1=pdf.to_numpy()
	#d2=gdf.to_numpy()
	print('Initialized!')
	
	params=[2,4,3,0.2,0.6,0.3,0.1,0.10,8,0.56] #infect-rate,onset-rate=0.88,Is-recovery-rate=0.56
	res=[]
	#args = sys.argv  #return arguments list to args
	'''
	M1,M2,M3,p1,p2,p3 = [
						float(args[1]),
					    float(args[2]),
						int(args[3]), 
						int(args[4]), 
						int(args[5]),
						int(args[6])
		  ]
	'''
    #args[0] is the name of py script, e.g. 'model.py'
	#step(24*54,3,param=params)
	step(24*54,4,params,pdf,gdf)
	
if __name__=="__main__":
	main()	