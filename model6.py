# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 06:37:09 2020

@author: mu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from py3plex.core import multinet
#from py3plex.core import random_generators
import matplotlib.image as mgimg
import pandas as pd
import networkx as nx
import param
import numpy.random as rnd
import geopandas as gpd
from datetime import datetime
import pymc3 as pm
#pdf['pos']=pdf.apply(lambda p: (gdf[gdf.gid==p.xy0].lat,gdf[gdf.gid==p.xy0].lon) ,axis=1)
########################################################
## real data
########################################################
def func(p,h):
	t_list=[i[0] for i in p.xy]
	lst=[abs(h-t) for t in t_list]
	idx=lst.index(min(lst))
	return int(p.xy[idx][1])
###########################################
pdf=pd.read_pickle(r'f:/_MU/pdf_24h.pkl')
gdf=pd.read_csv(r'f:/_MU/gid_xy.csv')
gdf.loc[gdf.Level2.isna(),'Level2']=101
df_y=pd.read_excel('./wh-2019-nCoV.xlsx')
y=df_y[df_y.city=='武汉市']['确诊'].to_numpy()[:31]

folder_tmp_files = './network/datasets/animation'
np.random.seed()

E_list=pdf.pid.sample(1)
pdf["County"] = pd.factorize(pdf["County"])[0].astype(np.int16)
pdf["group"] = pd.factorize(pdf["group"])[0].astype(np.int16)
pdf["state"] = pd.factorize(pdf["state"])[0].astype(np.int16)

pdf.loc[pdf.pid.isin(E_list),'state'] = 1  #0=S, 1=E
pdf['infected_duration']=0
#pdf['onset_duration']=8   #before 2-4 8 days,after 2 days
pdf['seviere_duration']=0
#pdf['hospital_duration']=0
#pdf['onset_duration']=0
pdf['native']=pdf.apply(lambda p: 0 if rnd.random() <= 0.357 else 1, axis=1)
l=len(pdf[pdf.native==0])
pdf['leave']=9999
pdf['leave'][pdf.native==0] = (rnd.random(l)*15).astype('int16')
#pdf['family_id']
pdf['Hospitalization']=0  #False
#pdf['ICU']=False
pdf['pos']=pdf.home_id	
pdf.iloc[:,list(range(5,29))+[34]]=pdf.iloc[:,list(range(5,29))+[34]].astype('int32')
pdf.iloc[:,list(range(29,34))]=pdf.iloc[:,list(range(29,34))].astype('int16')
#d1=pdf.to_numpy()
#d2=gdf.to_numpy()

print(pdf.info())
print('Initialized!')

############################

for tick in range(24*(55),24*(106),8):     #0,24*31,3
	#print(pInfect)
	#12.1-1.10封城前-春运开始
	#1.11-1.23封城， 学生放假
	#1.24-2.3
	#2.4-2.9
	#2.10-
	h = tick % 24
	place_in_risk=set()
	people_infected=pdf.loc[(pdf.state=='E')|(pdf.state=='Im')|(pdf.state=='Is')]
	place_in_risk=pdf.loc[(pdf.state=='E')|(pdf.state=='Im')|(pdf.state=='Is')].unique()
	#pdf.loc[pdf.state=='H']['pos']=gdf.loc[gdf.Level2==503]['gid'].sample(len(pdf.loc[pdf.state=='H']))
    
	for pir in place_in_risk:
		placeType=gdf.loc[gdf.gid==pir,'Level2'].values
		if placeType==101:
			n=np.random.randint(7) #row['family_size']
		elif placeType==503:
			n=np.random.randint(3)  #8
		else:
			n=np.random.randint(1)  #4
		#print(n)
		I_mask=(pdf.state.isin(['E','Im','Is']))&(pdf.pos==pir)
		S_mask=(pdf.state=='S')&(pdf.pos==pir)
		if pdf[S_mask].shape[0]>0:
			agentSet=pdf[S_mask].sample(min(n,pdf[S_mask].shape[0])).append(pdf[I_mask])
		else:
			agentSet=pdf[I_mask]
						
		N = len(agentSet)
		Ni = len(pdf[I_mask])	
			
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
		g=nx.empty_graph()

		#fig = plt.figure()
		#print(N)
		if N>Ni:
			if placeType==101:  # and (h > 5 and h < 23):
				M=min(N,params[0])    #m1=param[0]
				g = nx.watts_strogatz_graph(N, M, params[3], seed=None)
			elif placeType==503:
				M=min(N,params[1])
				g = nx.watts_strogatz_graph(N, M, params[4], seed=None)
			else:
				M=min(N,params[2])
				g = nx.watts_strogatz_graph(N, M, params[5], seed=None)
				#g = random_generators.random_multilayer_ER(N,floorNum,0.1,directed=False)
		

		#g = nx.watts_strogatz_graph(N, M, 0.2, seed=None)
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
		for i in g.nodes():			
			#传染beta U[0,1], 老人：N(3.14,0.1)
			if g.nodes[i]['state'] == 'E' or g.nodes[i]['state'] == 'Im' or g.nodes[i]['state'] == 'Is' or g.nodes[i]['state'] == 'H' :
				if g.nodes[i]['E_duration']>0:
					g.nodes[i]['E_duration']-= interval

				#S-E		
				for j in g.neighbors(i):
					if g.nodes[j]['state'] == 'S':
						if rnd.random() <= 0.025/interval:	   #pInfect
							g.nodes[j]['state'] = 'E'	
							pdf.loc[pdf.pid==j,'state'] = 'E' 
							#设定潜伏期							
							pdf.loc[pdf.pid==j,'E_duration']=(rnd.normal(5,1))*24/interval 
				#公交感染,根据24小时变化曲线
				#n=func(h)
			#潜伏期E-I   [3,7], N(5,1)				
			if g.nodes[i]['state'] == 'E' and g.nodes[i]['E_duration']<=0:   #潜伏期结束
				#潜伏转有症状(I or Is)
				if rnd.random() <= 0.9:
					if rnd.random() <= 0.22:
						g.nodes[i]['Im2s_duration']=rnd.normal(5,1)*24/interval
					else:
						g.nodes[i]['Im2s_duration']=9999

					pdf.loc[pdf.pid==i,'I_duration']=(rnd.normal(17,3))*24/interval
				#潜伏转无症状感染(,不传染，不统计)
				else:
				    g.nodes[i]['state'] = 'In'
					pdf.loc[pdf.pid==i,'state'] = 'In' 
					
			#轻症		  
			if g.nodes[i]['state'] == 'Im':
				#轻症转重症 N(5,1),22%
				if g.nodes[i]['Im2s_duration']<= 0:    #after 15day, became s   #0.013 * 3 :    #0.013
					g.nodes[i]['state'] = 'Is'
				#轻症确诊N(14,2^2), ->恢复
				if rnd.random() <= 0.5:  #确诊率50%
					#g.nodes[i]['state'] = 'Imc'
					#pdf.loc[pdf.pid==i,'state']='Imc'
					if rnd.random() <= 0.5:  #收治率， 床位数
						#确诊，在医院恢复
						pdf.loc[pdf.pid==i,'I_duration']=(rnd.normal(14,2))*24/interval 	
					##else:
						#确诊，在家恢复
						#pdf.loc[pdf.pid==i,'I_duration']=(rnd.normal(17,3))*24/interval
				
				#轻症未确诊在家	N(22,3)
				##else:	 
					#pdf.loc[pdf.pid==i,'recovery_duration']=(rnd.normal(17,3))*24/interval
			#重症			
			if g.nodes[i]['state'] == 'Is' 
				if g.nodes[i]['I_duration']>0:
					#重症确诊					
					if rnd.random() <= 0.8:  #确诊率80%
						#g.nodes[i]['state'] = 'Isc'
						#pdf.loc[pdf.pid==i,'state']='Isc'
						if rnd.random() <= 0.8:  #重症收治率， 床位数
							#重症收治恢复50% N(17,3)				
							if rnd.random()<0.5:
								pdf.loc[pdf.pid==i,'state']='Is2r'
							'''	
							#重症收治死亡50% N(17,3)
							else:
								pdf.loc[pdf.pid==i,'state']='R'
							'''
						else:
							#重症确诊未收治在家恢复 25% N(17,3)				
							if rnd.random()<0.25:
								pdf.loc[pdf.pid==i,'state']='Is2r'
							'''
							#重症确诊未收治在家死亡75% N(17,3)
							else:
								pdf.loc[pdf.pid==i,'state']='R'		
							'''
					#重症未确诊，1-0.8=0.2,	N(22,3^3)
					else:	 
						#重症未确诊死亡100%
						pdf.loc[pdf.pid==i,'state']='Is2d'
				else:
					pdf.loc[pdf.pid==i,'state']='R'
			if g.nodes[i]['state'] == 'Is2d' and g.nodes[i]['Is_duration']<= 0:
				pdf.loc[pdf.pid==i,'state']='D'
                  	 

                    
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
	if h==0:
		sNum=len(pdf.loc[pdf.state == 'S'])
		eNum=len(pdf.loc[pdf.state == 'E'])
		iNum=len(pdf.loc[pdf.state == 'I'])        
		isNum=len(pdf.loc[pdf.state == 'Is'])
		hNum=len(pdf.loc[pdf.state == 'H'])    
		rNum=len(pdf.loc[pdf.state == 'R'])
		dNum=len(pdf.loc[pdf.state == 'D']) 
		
		ts=datetime.now().strftime('%Y.%m.%d-%H:%M:%S')       
		l=(tick/24+1,sNum,eNum,iNum,isNum,hNum,rNum,dNum,ts)  
	
		res.append(l)
		print(l)

df_res=pd.DataFrame(res) 
df_res.columns=['day','S','E','I','Is','H','R','D','ts']   

import time
df_res.to_csv(r'f:/_MU/res_2019_by_2020_1_23'+str(int(time.time()))+'.csv')  
pdf.to_pickle(r'f:/_MU/pdf_2020_1_20.pkl')