-- 2 Simples

-- Mostre nomes e cagos de todos os usuarios
SELECT nome, cargo FROM Usuario;

-- Mostre os dados dos armazens em ordem crescente de capacidade
SELECT * FROM Armazem ORDER BY capacidade;


-- 3 Junções internas

-- Nome dos controladores que alocaram cada Recurso de plantio
SELECT nome, idRecPlantio FROM 
Controlador_aloca_RecPlantio INNER JOIN Usuario
ON Controlador_aloca_RecPlantio.idControlador = Usuario.idUsuario;

-- Mostre os id's das tarefas de plantio com o nome do usuario responsavel
SELECT idTarefa, nome FROM 
TrfPlantio INNER JOIN Usuario
ON idOperador = idUsuario;

-- para cada colheita, mostre o nome da cultura que foi colhida, a producao, com qual modelo de colheitadeira,
-- o gasto total de combustivel para a colheita (sem gasto é vazio),
-- o nome do operador e um telefone dele (se tiver), local do armazem onde foi estocada, nome do controlador desse armazem,
-- sua formação, a data do plantio e da colheita, e o local que ocorreu essa colheita (local do terreno),
-- apenas das colheitas que tiveram produção maior que 1000, e em ordem decrescente de produção
DROP VIEW combust_colheitas;
CREATE VIEW combust_colheitas (uso_total, idPlantio, seq_colheita) AS
    SELECT SUM(RCE.qtd), RCE.idPlantio, RCE.seq_colheita
    FROM RecColheita_espc AS RCE
    INNER JOIN Combust_colheita AS comb ON RCE.idRecColheita = comb.idRecColheita
    GROUP BY RCE.idPlantio, RCE.seq_colheita;
SELECT
Cult.nome AS Cultura,
Col.producao AS Produção,
Eq.modelo AS Colheitadeira,
CC.uso_total AS Uso_de_combustível,
Op.nome AS Operador,
MAX(Tel.Fone) AS Fone_do_Operador,
Arm.local AS Local_de_estoque,
CtrlUsu.nome AS Controlador_do_armazém,
Ctrl.formacao AS Formacao_controlador,
Pl.data AS Data_do_plantio,
Col.data AS Data_da_colheita,
Terr.local AS Terreno_Colhido
FROM Colheita AS Col
INNER JOIN Equipamento AS Eq ON Eq.idEquipamento = Col.idColheitadeira
INNER JOIN Plantio AS Pl ON Pl.idPlantio = Col.idPlantio
INNER JOIN Armazem AS Arm ON Arm.idArmazem = Col.idArmazem
INNER JOIN Usuario AS Op ON Op.idUsuario = Col.idOperador
LEFT JOIN Fone_Usuario AS Tel ON Tel.idUsuario = Col.idOperador
INNER JOIN Cultura AS Cult ON Cult.idCultura = Pl.idCultura
LEFT JOIN combust_colheitas AS CC ON CC.idPlantio = Col.idPlantio
  AND CC.seq_colheita = Col.seq_colheita
INNER JOIN Usuario AS CtrlUsu ON CtrlUsu.idUsuario = Arm.idControlador
INNER JOIN Controlador AS Ctrl ON Ctrl.idUsuario = Arm.idControlador
INNER JOIN Terreno AS Terr ON Terr.idTerreno = Pl.idTerreno
WHERE Col.producao > 1000
GROUP BY Cultura, Produção, Colheitadeira, Operador, Local_de_estoque, Controlador_do_armazém,
  Formacao_controlador, Data_do_plantio, Data_da_colheita, Terreno_Colhido, Uso_de_combustível
ORDER BY Col.producao DESC;

-- 1 Junção externa

-- dois telefones de cada usuario
SELECT U.nome AS Nome_Usuario, MIN(Tel1.Fone) AS Fone1, MAX(Tel2.Fone) AS Fone2
FROM Usuario U LEFT OUTER JOIN Fone_Usuario Tel1 ON U.idUsuario = Tel1.idUsuario
LEFT OUTER JOIN Fone_Usuario Tel2 ON U.idUsuario = Tel2.idUsuario AND Tel1.Fone != Tel2.Fone
GROUP BY U.idUsuario;

-- 3 Agrupamentos

-- total de horas trabalhadas por cada cargo
SELECT SUM(hrs_trab), cargo
FROM Usuario
GROUP BY cargo;

-- armazem com maior capacidade de cada local
SELECT MAX(capacidade), local
FROM Armazem
GROUP BY local;

-- uso medio entre os diferentes combustiveis de cada colheita
SELECT ROUND(AVG(RCE.qtd), 2), RCE.idPlantio, RCE.seq_colheita
FROM RecColheita_espc AS RCE
INNER JOIN Combust_colheita AS comb ON RCE.idRecColheita = comb.idRecColheita
GROUP BY RCE.idPlantio, RCE.seq_colheita;

-- 1 Subconsulta

-- todas as pragas registradas que possuem um inseticida que a combata registrado
SELECT DISTINCT Praga FROM Praga_Cultura
WHERE Praga IN (SELECT combat_praga FROM Inseticida);
