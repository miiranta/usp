#Trabalho 2 - Grafos
#Lucas Miranda Mendonça Rezende
#12542838

import networkx as nx
import os
import matplotlib.pyplot as plt

#Criando grafo DIRECIONADO 
DG_all = nx.DiGraph()
DG1 = nx.DiGraph() #Subgrafo tipo 1
DG2 = nx.DiGraph() #Subgrafo tipo 2
DG3 = nx.DiGraph() #Subgrafo tipo 3

#Abrindo o arquivo de descrição dos papers
rel_path = "pubmed-diabetes/data/Pubmed-Diabetes.NODE.paper.tab"
script_dir = os.path.dirname(__file__)
abs_file_path = os.path.join(script_dir, rel_path)

f_desc = open(abs_file_path, "r")
lines_desc = f_desc.readlines()

#Tirando as primeiras linhas do arquivo
lines_desc.remove(lines_desc[1])
lines_desc.remove(lines_desc[0])

#Criando nós do grafo com o tipo de DM
for line in lines_desc:
    line_content = line.split()
    
    ref1 = line_content[0].strip()
    ref2 = line_content[1].split("=")[1].strip()
    
    DG_all.add_node(int(ref1), title=int(ref2))
    
    if(int(ref2) == 1):
        DG1.add_node(int(ref1), title=int(ref2))
    elif(int(ref2) == 2):
        DG2.add_node(int(ref1), title=int(ref2))
    elif(int(ref2) == 3):
        DG3.add_node(int(ref1), title=int(ref2))

#Abrindo o arquivo de citações
rel_path = "pubmed-diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab"
script_dir = os.path.dirname(__file__)
abs_file_path = os.path.join(script_dir, rel_path)

f_cites = open(abs_file_path, "r")
lines_cites = f_cites.readlines()

#Tirando as primeiras linhas do arquivo
lines_cites.remove(lines_cites[1])
lines_cites.remove(lines_cites[0])

#Para cada linha, pegar os números de referência dos papers e adicionar como aresta do grafo
all_nodes = dict(list(DG_all.nodes.data("title")))
for line in lines_cites:

    line_content = line.split()

    ref1 = line_content[1].split(":")[1].strip()
    ref2 = line_content[3].split(":")[1].strip()

    DG_all.add_edge(int(ref1), int(ref2))

    if(all_nodes[int(ref1)] == 1 and all_nodes[int(ref2)] == 1):
        DG1.add_edge(int(ref1), int(ref2))
    elif(all_nodes[int(ref1)] == 2 and all_nodes[int(ref2)] == 2):
        DG2.add_edge(int(ref1), int(ref2))
    elif(all_nodes[int(ref1)] == 3 and all_nodes[int(ref2)] == 3):
        DG3.add_edge(int(ref1), int(ref2))

#Grafos/subgrafos e suas informações
print("DG_all |", "Nós:", DG_all.number_of_nodes(), "| Arestas:", DG_all.number_of_edges())
print("DG1 |", "Nós:", DG1.number_of_nodes(), "| Arestas:", DG1.number_of_edges())
print("DG2 |", "Nós:", DG2.number_of_nodes(), "| Arestas:", DG2.number_of_edges())
print("DG3 |", "Nós:", DG3.number_of_nodes(), "| Arestas:", DG3.number_of_edges())



#DESAFIO 2
print("\n\nDesafio 2")

#1.1. Qual o inDegree e outDegree total dos grafos
print("\n1.1. Qual o inDegree e outDegree total dos grafos")
print("DG_all |", "inDegree:", sum(dict(DG_all.in_degree()).values()), "| outDegree:", sum(dict(DG_all.out_degree()).values()))
print("DG1 |", "inDegree:", sum(dict(DG1.in_degree()).values()), "| outDegree:", sum(dict(DG1.out_degree()).values()))
print("DG2 |", "inDegree:", sum(dict(DG2.in_degree()).values()), "| outDegree:", sum(dict(DG2.out_degree()).values()))
print("DG3 |", "inDegree:", sum(dict(DG3.in_degree()).values()), "| outDegree:", sum(dict(DG3.out_degree()).values()))

#1.2. Os 3 vértices com maiores inDegree e outDegree de cada grafo
print("\n1.2. Os 3 vértices com maiores inDegree e outDegree de cada grafo")
print("DG_all ", "\n -inDegree:", sorted(dict(DG_all.in_degree()).items(), key=lambda x: x[1], reverse=True)[:3], "\n -outDegree:", sorted(dict(DG_all.out_degree()).items(), key=lambda x: x[1], reverse=True)[:3])
print("DG1 ", "\n -inDegree:", sorted(dict(DG1.in_degree()).items(), key=lambda x: x[1], reverse=True)[:3], "\n -outDegree:", sorted(dict(DG1.out_degree()).items(), key=lambda x: x[1], reverse=True)[:3])
print("DG2 ", "\n -inDegree:", sorted(dict(DG2.in_degree()).items(), key=lambda x: x[1], reverse=True)[:3], "\n -outDegree:", sorted(dict(DG2.out_degree()).items(), key=lambda x: x[1], reverse=True)[:3])
print("DG3 ", "\n -inDegree:", sorted(dict(DG3.in_degree()).items(), key=lambda x: x[1], reverse=True)[:3], "\n -outDegree:", sorted(dict(DG3.out_degree()).items(), key=lambda x: x[1], reverse=True)[:3])

#2. Calcule a reciprocidade do vértice com maior inDegree e do vértice com maior outDegree
print("\n2. Reciprocidade do vértice com maior inDegree e do vértice com maior outDegree")
DG_biggest_inDegree = sorted(dict(DG_all.in_degree()).items(), key=lambda x: x[1], reverse=True)[0][0]
DG_biggest_outDegree = sorted(dict(DG_all.out_degree()).items(), key=lambda x: x[1], reverse=True)[0][0]
DG_inDegree_reciprocity = nx.reciprocity(DG_all, nodes=DG_biggest_inDegree)
DG_outDegree_reciprocity = nx.reciprocity(DG_all, nodes=DG_biggest_outDegree)
print("DG_all")
print(" -Maior inDegree:", DG_biggest_inDegree, "| Reciprocidade:", DG_inDegree_reciprocity)
print(" -Maior outDegree:", DG_biggest_outDegree, "| Reciprocidade:", DG_outDegree_reciprocity)

DG1_biggest_inDegree = sorted(dict(DG1.in_degree()).items(), key=lambda x: x[1], reverse=True)[0][0]
DG1_biggest_outDegree = sorted(dict(DG1.out_degree()).items(), key=lambda x: x[1], reverse=True)[0][0]
DG1_inDegree_reciprocity = nx.reciprocity(DG1, nodes=DG1_biggest_inDegree)
DG1_outDegree_reciprocity = nx.reciprocity(DG1, nodes=DG1_biggest_outDegree)
print("DG1")
print(" -Maior inDegree:", DG1_biggest_inDegree, "| Reciprocidade:", DG1_inDegree_reciprocity)
print(" -Maior outDegree:", DG1_biggest_outDegree, "| Reciprocidade:", DG1_outDegree_reciprocity)

DG2_biggest_inDegree = sorted(dict(DG2.in_degree()).items(), key=lambda x: x[1], reverse=True)[0][0]
DG2_biggest_outDegree = sorted(dict(DG2.out_degree()).items(), key=lambda x: x[1], reverse=True)[0][0]
DG2_inDegree_reciprocity = nx.reciprocity(DG2, nodes=DG2_biggest_inDegree)
DG2_outDegree_reciprocity = nx.reciprocity(DG2, nodes=DG2_biggest_outDegree)
print("DG2")
print(" -Maior inDegree:", DG2_biggest_inDegree, "| Reciprocidade:", DG2_inDegree_reciprocity)
print(" -Maior outDegree:", DG2_biggest_outDegree, "| Reciprocidade:", DG2_outDegree_reciprocity)

DG3_biggest_inDegree = sorted(dict(DG3.in_degree()).items(), key=lambda x: x[1], reverse=True)[0][0]
DG3_biggest_outDegree = sorted(dict(DG3.out_degree()).items(), key=lambda x: x[1], reverse=True)[0][0]
DG3_inDegree_reciprocity = nx.reciprocity(DG3, nodes=DG3_biggest_inDegree)
DG3_outDegree_reciprocity = nx.reciprocity(DG3, nodes=DG3_biggest_outDegree)
print("DG3")
print(" -Maior inDegree:", DG3_biggest_inDegree, "| Reciprocidade:", DG3_inDegree_reciprocity)
print(" -Maior outDegree:", DG3_biggest_outDegree, "| Reciprocidade:", DG3_outDegree_reciprocity)

#3. Calcule a reciprocidade de todo o grafo
print("\n3. Reciprocidade de todo o grafo")
print("DG_all |", nx.overall_reciprocity(DG_all))
print("DG1 |", nx.overall_reciprocity(DG1))
print("DG2 |", nx.overall_reciprocity(DG2))
print("DG3 |", nx.overall_reciprocity(DG3))

#4. Calcule o pagerank dos 3 vértices mais citados de cada grafo
print("\n4. Pagerank dos 3 vértices mais citados de cada grafo")

DG_pageRank = nx.pagerank(DG_all, max_iter=1000, alpha=0.85, tol=1e-06, nstart=None, weight='weight', dangling=None)
print("DG_all |", sorted(DG_pageRank.items(), key=lambda x: x[1], reverse=True)[:3])

DG1_pageRank = nx.pagerank(DG1, max_iter=1000, alpha=0.85, tol=1e-06, nstart=None, weight='weight', dangling=None)
print("DG1 |", sorted(DG1_pageRank.items(), key=lambda x: x[1], reverse=True)[:3])

DG2_pageRank = nx.pagerank(DG2, max_iter=1000, alpha=0.85, tol=1e-06, nstart=None, weight='weight', dangling=None)
print("DG2 |", sorted(DG2_pageRank.items(), key=lambda x: x[1], reverse=True)[:3])

DG3_pageRank = nx.pagerank(DG3, max_iter=1000, alpha=0.85, tol=1e-06, nstart=None, weight='weight', dangling=None)
print("DG3 |", sorted(DG3_pageRank.items(), key=lambda x: x[1], reverse=True)[:3])

