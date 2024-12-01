#Trabalho 2 - Grafos
#Lucas Miranda Mendonça Rezende
#12542838

import networkx as nx
import os
import matplotlib.pyplot as plt

#Abrindo o arquivo de citações
rel_path = "pubmed-diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab"
script_dir = os.path.dirname(__file__)
abs_file_path = os.path.join(script_dir, rel_path)

f_cites = open(abs_file_path, "r")
lines_cites = f_cites.readlines()

#Tirando as primeiras linhas do arquivo
lines_cites.remove(lines_cites[1])
lines_cites.remove(lines_cites[0])

#Criando grafo NÃO DIRECIONADO
G = nx.Graph()

#Criando grafo DIRECIONADO 
DG = nx.DiGraph()

#Para cada linha, pegar os números de referência dos papers e adicionar como aresta do grafo
for line in lines_cites:

    line_content = line.split()

    ref1 = line_content[1].split(":")[1].strip()
    ref2 = line_content[3].split(":")[1].strip()

    G.add_edge(int(ref1), int(ref2))
    DG.add_edge(int(ref1), int(ref2))



#DESAFIO 1
print("Desafio 1")

#1. Némero de nós e arestas
print("1. Número de nós:", G.number_of_nodes() ,"| Número de arestas:", G.number_of_edges())

#2. Grau mínimo, médio e máximo
print("2. Grau mínimo:", min(dict(G.degree()).values()), "| Grau médio:", sum(dict(G.degree()).values())/len(dict(G.degree()).values()), "| Grau máximo:", max(dict(G.degree()).values()))

#3. Densidade da rede
print("3. Densidade da rede:", nx.density(G))

#4. Número médio de triângulos
number_of_triangles = sum(nx.triangles(G).values()) / 3
avarage_number_of_triangles = number_of_triangles / G.number_of_nodes()
print("4. Número médio de triângulos:", avarage_number_of_triangles)

#5. Média do coeficiente de agrupamento (clustering)
print("5. Coeficiente de agrupamento médio:", nx.average_clustering(G))

#6. Diâmetro da rede
print("6. Diametro da rede (aproximado):", nx.approximation.diameter(G)) #17 ou 18, sem usar aproximação fica extremamente lento

#7. Número de componentes conexos fracos, além do número de nós e arestas para o maior destes componentes
print("7. Número de componentes conexos fracos:", nx.number_weakly_connected_components(DG), "| Número de nós:", max([len(c) for c in sorted(nx.weakly_connected_components(DG), key=len, reverse=True)]), "| Número de arestas:", max([len(c) for c in sorted(nx.weakly_connected_components(DG), key=len, reverse=True)]))

#8. Número de componentes conexos fortes, além do número de nós e arestas para o maior destes componentes
print("8. Número de componentes conexos fortes:", nx.number_strongly_connected_components(DG), "| Número de nós:", max([len(c) for c in sorted(nx.strongly_connected_components(DG), key=len, reverse=True)]), "| Número de arestas:", max([len(c) for c in sorted(nx.strongly_connected_components(DG), key=len, reverse=True)]))

#9. Elencar os 5 vértices de maior grau
print("9. Os 5 vértices de maior grau:", sorted(dict(G.degree()).items(), key=lambda x: x[1], reverse=True)[:5])

#10. Elucidar o vértice de maior centralidade de intermediação (betweenness centrality)
central_bet = sorted(nx.betweenness_centrality(G, k=100).items(), key=lambda x: x[1], reverse=True)[:1]
print("10. O vértice de maior centralidade de intermediação (aproximado):", central_bet)

#11. O vértice de maior eigenvector centrality, ou centralidade do autovetor.
central_vector = sorted(nx.eigenvector_centrality(G).items(), key=lambda x: x[1], reverse=True)[:1]
print("11. O vértice de maior centralidade do autovetor:", central_vector)

#Criando uma representação aproximada do grafo
DrawMe = nx.Graph()

vert_maior_grau_quant = 20 #Quantos vértices de maior grau serão desenhados
edge_limit = 50 #Quantas arestas serão consideradas para cada vértice

vert_maior_grau = sorted(dict(G.degree()).items(), key=lambda x: x[1], reverse=True)[:vert_maior_grau_quant]

for vert in vert_maior_grau:
    DrawMe.add_node(vert[0], size=vert[1]*10)
    
    count = 0
    for edge in G.edges(vert[0]):
        if(count>=edge_limit):
            break
        count+=1
        
        if(edge[0] not in DrawMe.nodes):
            DrawMe.add_node(edge[0], size=20)
            
        if(edge[1] not in DrawMe.nodes):
            DrawMe.add_node(edge[1], size=20)
        
        DrawMe.add_edge(edge[0], edge[1])

print("\nFazendo representação aproximada do grafo")
print("Número de nós principais considerados:", vert_maior_grau_quant)
print("Número de arestas consideradas por nó:", edge_limit)
print("Número de nós total do grafo:", DrawMe.number_of_nodes())
print("Número de arestas do grafo:", DrawMe.number_of_edges())

pos = nx.spring_layout(DrawMe, scale=100, k=0.22, iterations=120) #Configurações de plotagem

print("Plotando...")
nx.draw(DrawMe, pos, node_size=[v[1] for v in DrawMe.nodes.data('size')], with_labels=True, font_size=5)
plt.show() 
