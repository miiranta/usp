-- Equipamento

    -- Colheitadeira
    INSERT INTO Equipamento
    VALUES (nextval('idEquipamento_seq'), 'GTX-730', '2020-07-09', 'Colheitadeira');
    INSERT INTO Colheitadeira
    VALUES (12, currval('idEquipamento_seq'));
    INSERT INTO Equipamento
    VALUES (nextval('idEquipamento_seq'), 'MX330', '2018-01-03', 'Colheitadeira');
    INSERT INTO Colheitadeira
    VALUES (7, currval('idEquipamento_seq'));

    -- Trator
    INSERT INTO Equipamento
    VALUES (nextval('idEquipamento_seq'), 'RTX-3060', '2020-07-09', 'Trator');
    INSERT INTO Trator
    VALUES (12000, currval('idEquipamento_seq'));
    INSERT INTO Equipamento
    VALUES (nextval('idEquipamento_seq'), 'RX6600', '2020-10-09', 'Trator');
    INSERT INTO Trator
    VALUES (8000, currval('idEquipamento_seq'));

-- Usuarios, espec e fones

    -- AdmPlantio
    INSERT INTO Usuario
    VALUES (nextval('idUsuario_seq'), 'Marcin', 2486, 'senha123', 'AdmPlantio');
    INSERT INTO AdmPlantio (formacao, idUsuario) VALUES ('UFMG', currval('idUsuario_seq'));
    INSERT INTO Fone_Usuario (Fone, idUsuario)
    VALUES ('11928174921', currval('idUsuario_seq'));
    INSERT INTO Fone_Usuario (Fone, idUsuario)
    VALUES ('19923274625', currval('idUsuario_seq'));
    INSERT INTO Fone_Usuario (Fone, idUsuario)
    VALUES ('16928374625', currval('idUsuario_seq'));

    -- Agronomo
    INSERT INTO Usuario
    VALUES (nextval('idUsuario_seq'), 'Juan', 999, 'senha123', 'Agronomo');
    INSERT INTO Agronomo (formacao, terreno_atuac, idUsuario) VALUES ('EXALQ', 'varios terrenos', currval('idUsuario_seq'));
    INSERT INTO Fone_Usuario (Fone, idUsuario)
    VALUES ('19928274725', currval('idUsuario_seq'));
    INSERT INTO Fone_Usuario (Fone, idUsuario)
    VALUES ('17922371625', currval('idUsuario_seq'));
    INSERT INTO Fone_Usuario (Fone, idUsuario)
    VALUES ('12928374625', currval('idUsuario_seq'));

    -- Operador
    INSERT INTO Usuario
    VALUES (nextval('idUsuario_seq'), 'Mogu Mogu', 1111, 'mug', 'Operador');
    INSERT INTO Operador (desempenho, equipe, idUsuario, espec) VALUES (9, 3, currval('idUsuario_seq'), 'Moguniversity');
    INSERT INTO Fone_Usuario (Fone, idUsuario)
    VALUES ('11111111111', currval('idUsuario_seq'));
    INSERT INTO Fone_Usuario (Fone, idUsuario)
    VALUES ('17122171611', currval('idUsuario_seq'));
    INSERT INTO Fone_Usuario (Fone, idUsuario)
    VALUES ('12928474611', currval('idUsuario_seq'));
    INSERT INTO Usuario
    VALUES (nextval('idUsuario_seq'), 'Amomogusgus', 3333, 'sus', 'Operador');
    INSERT INTO Operador (desempenho, equipe, idUsuario, espec) VALUES (11, 3, currval('idUsuario_seq'), 'Moguniversity');
    INSERT INTO Fone_Usuario (Fone, idUsuario)
    VALUES ('17171717171', currval('idUsuario_seq'));

    -- Controlador
    INSERT INTO Usuario (idUsuario, nome, hrs_trab, senha, cargo)
    VALUES (nextval('idUsuario_seq'), 'Jorge', 1286, 'senha123', 'Controlador');
    INSERT INTO Controlador (formacao, desempenho, idUsuario, espec)
    VALUES ('Técnico em controle', 7, currval('idUsuario_seq'), 'Armazém');
    INSERT INTO Fone_Usuario (Fone, idUsuario)
    VALUES ('16978324925', currval('idUsuario_seq'));
    INSERT INTO Fone_Usuario (Fone, idUsuario)
    VALUES ('17923456677', currval('idUsuario_seq'));
    INSERT INTO Fone_Usuario (Fone, idUsuario)
    VALUES ('12987564544', currval('idUsuario_seq'));

    -- AdmSistema
    INSERT INTO Usuario
    VALUES (nextval('idUsuario_seq'), 'Amogus', 9999, 'sus', 'AdmSistema');
    INSERT INTO AdmSistema (nv_acesso, idUsuario) VALUES (5, currval('idUsuario_seq'));
    INSERT INTO Fone_Usuario (Fone, idUsuario)
    VALUES ('99999999999', currval('idUsuario_seq'));
    INSERT INTO Fone_Usuario (Fone, idUsuario)
    VALUES ('88888888888', currval('idUsuario_seq'));
    INSERT INTO Fone_Usuario (Fone, idUsuario)
    VALUES ('77777777777', currval('idUsuario_seq'));

-- Armazem
INSERT INTO Armazem
VALUES (nextval('idArmazem_seq'), 'Lote 11', 100, (SELECT MAX(idUsuario) FROM Controlador));
INSERT INTO Armazem 
VALUES (nextval('idArmazem_seq'), 'Lote 22', 200, (SELECT MAX(idUsuario) FROM Controlador));
INSERT INTO Armazem 
VALUES (nextval('idArmazem_seq'), 'Lote 33', 400, (SELECT MAX(idUsuario) FROM Controlador));
INSERT INTO Armazem 
VALUES (nextval('idArmazem_seq'), 'Lote 44', 300, (SELECT MAX(idUsuario) FROM Controlador));
INSERT INTO Armazem 
VALUES (nextval('idArmazem_seq'), 'Lote 55', 230, (SELECT MAX(idUsuario) FROM Controlador));

-- RecPlantio

    -- Combust_plantio
    INSERT INTO RecPlantio
    VALUES (nextval('idRecPlantio_seq'), 'Oleo', 'Shell', 'Combust_plantio', (SELECT MAX(idArmazem) FROM Armazem));
    INSERT INTO Combust_plantio
    VALUES ('Diesel', currval('idRecPlantio_seq'));

    -- Inseticida
    INSERT INTO RecPlantio
    VALUES (nextval('idRecPlantio_seq'), 'Mata-inseto', 'Raid', 'Inseticida', (SELECT MAX(idArmazem)-1 FROM Armazem));
    INSERT INTO Inseticida
    VALUES ('Minhoca', currval('idRecPlantio_seq'));

    -- Semente
    INSERT INTO RecPlantio
    VALUES (nextval('idRecPlantio_seq'), 'Feijon', 'Agrotop', 'Semente', (SELECT MAX(idArmazem)-2 FROM Armazem));
    INSERT INTO Semente
    VALUES ('Eucarionte', currval('idRecPlantio_seq'));

    -- Fertiliz
    INSERT INTO RecPlantio
    VALUES (nextval('idRecPlantio_seq'), 'Fértil', 'Agrotop', 'Fertiliz', (SELECT MAX(idArmazem)-3 FROM Armazem));
    INSERT INTO Fertiliz
    VALUES ('nitro', 'fosforo', 'pota', currval('idRecPlantio_seq'));

-- RecColheita

    -- Combust_colheita
    INSERT INTO RecColheita
    VALUES (nextval('idRecColheita_seq'), 'Oleo', 'Shell', 'Combust_colheita', (SELECT MAX(idArmazem) FROM Armazem));
    INSERT INTO Combust_colheita
    VALUES ('Diesel', currval('idRecColheita_seq'));
    INSERT INTO RecColheita
    VALUES (nextval('idRecColheita_seq'), 'Gasolina', 'Ipiranga', 'Combust_colheita', (SELECT MAX(idArmazem) FROM Armazem));
    INSERT INTO Combust_colheita
    VALUES ('Não-aditivada', currval('idRecColheita_seq'));

    -- Saca
    INSERT INTO RecColheita
    VALUES (nextval('idRecColheita_seq'), 'Saca de feijao', 'SacasMil', 'Saca', (SELECT MAX(idArmazem)-2 FROM Armazem));
    INSERT INTO Saca
    VALUES (20, 'Trapo', currval('idRecColheita_seq'));

-- Terreno
INSERT INTO Terreno
VALUES (nextval('idTerreno_seq'), 'Lote 12', 'Bonito hein', 100, 0.5, 0.5, 0.5, (SELECT MAX(idUsuario) FROM AdmPlantio));
INSERT INTO Terreno
VALUES (nextval('idTerreno_seq'), 'Lote 22', 'Bonito'     , 150, 0.5, 0.5, 0.5, (SELECT MAX(idUsuario) FROM AdmPlantio));
INSERT INTO Terreno
VALUES (nextval('idTerreno_seq'), 'Lote 32', 'Marromenos' , 300, 0.5, 0.5, 0.5, (SELECT MAX(idUsuario) FROM AdmPlantio));
INSERT INTO Terreno
VALUES (nextval('idTerreno_seq'), 'Lote 42', 'Feio'       , 50 , 0.5, 0.5, 0.5, (SELECT MAX(idUsuario) FROM AdmPlantio));
INSERT INTO Terreno
VALUES (nextval('idTerreno_seq'), 'Lote 52', 'Feio hein'  , 200, 0.5, 0.5, 0.5, (SELECT MAX(idUsuario) FROM AdmPlantio));

-- Cultura e suas pragas
INSERT INTO Cultura
VALUES (nextval('idCultura_seq'), 'Cultura 1', 'Agriao'    , 10, (SELECT MAX(idUsuario) FROM Agronomo));
INSERT INTO Praga_Cultura
VALUES ('Minhoca', currval('idCultura_seq'));
INSERT INTO Praga_Cultura
VALUES ('Besouro', currval('idCultura_seq'));
INSERT INTO Cultura
VALUES (nextval('idCultura_seq'), 'Cultura 2', 'Manjericao', 20, (SELECT MAX(idUsuario) FROM Agronomo));
INSERT INTO Praga_Cultura
VALUES ('Minhoca', currval('idCultura_seq'));
INSERT INTO Praga_Cultura
VALUES ('Besouro', currval('idCultura_seq'));
INSERT INTO Cultura
VALUES (nextval('idCultura_seq'), 'Cultura 3', 'Alface'    , 20, (SELECT MAX(idUsuario) FROM Agronomo));
INSERT INTO Praga_Cultura
VALUES ('Minhoca', currval('idCultura_seq'));
INSERT INTO Praga_Cultura
VALUES ('Javali', currval('idCultura_seq'));
INSERT INTO Cultura
VALUES (nextval('idCultura_seq'), 'Cultura 4', 'Tomate'    , 30, (SELECT MAX(idUsuario) FROM Agronomo));
INSERT INTO Praga_Cultura
VALUES ('Javali', currval('idCultura_seq'));
INSERT INTO Cultura
VALUES (nextval('idCultura_seq'), 'Cultura 5', 'Bacon'     , 10, (SELECT MAX(idUsuario) FROM Agronomo));
INSERT INTO Praga_Cultura
VALUES ('Licia', currval('idCultura_seq'));
INSERT INTO Praga_Cultura
VALUES ('Miranda', currval('idCultura_seq'));

-- Plantio
INSERT INTO Plantio
VALUES (nextval('idPlantio_seq'), '2020-01-01', (SELECT MAX(idCultura) FROM Cultura), (SELECT MAX(idTerreno) FROM Terreno));
INSERT INTO Plantio
VALUES (nextval('idPlantio_seq'), '2021-01-01', (SELECT MAX(idCultura)-1 FROM Cultura), (SELECT MAX(idTerreno)-1 FROM Terreno));
INSERT INTO Plantio
VALUES (nextval('idPlantio_seq'), '2022-01-01', (SELECT MAX(idCultura)-2 FROM Cultura), (SELECT MAX(idTerreno)-2 FROM Terreno));
INSERT INTO Plantio
VALUES (nextval('idPlantio_seq'), '2023-01-01', (SELECT MAX(idCultura)-3 FROM Cultura), (SELECT MAX(idTerreno)-3 FROM Terreno));
INSERT INTO Plantio
VALUES (nextval('idPlantio_seq'), '2024-01-01', (SELECT MAX(idCultura)-4 FROM Cultura), (SELECT MAX(idTerreno)-4 FROM Terreno));

-- Colheita
INSERT INTO Colheita
VALUES (
 1000000000, 1,
 (SELECT MAX(idEquipamento)-1 FROM Colheitadeira),
 (SELECT MAX(idPlantio)-4 FROM Plantio), '2020-07-01',
 (SELECT MAX(idArmazem) FROM Armazem),
 (SELECT MAX(idUsuario) FROM Operador)
);
INSERT INTO Colheita
VALUES (
 12345, 1,
 (SELECT MAX(idEquipamento)-1 FROM Colheitadeira),
 (SELECT MAX(idPlantio)-3 FROM Plantio), '2021-07-01',
 (SELECT MAX(idArmazem)-1 FROM Armazem),
 (SELECT MAX(idUsuario) FROM Operador)
);
INSERT INTO Colheita
VALUES (
 10000, 1,
 (SELECT MAX(idEquipamento)-1 FROM Colheitadeira),
 (SELECT MAX(idPlantio)-2 FROM Plantio), '2022-07-01',
 (SELECT MAX(idArmazem)-2 FROM Armazem),
 (SELECT MAX(idUsuario)-1 FROM Operador)
);
INSERT INTO Colheita
VALUES (
 8241, 1,
 (SELECT MAX(idEquipamento) FROM Colheitadeira),
 (SELECT MAX(idPlantio)-1 FROM Plantio), '2023-07-01',
 (SELECT MAX(idArmazem)-3 FROM Armazem),
 (SELECT MAX(idUsuario)-1 FROM Operador)
);
INSERT INTO Colheita
VALUES (
 10, 1,
 (SELECT MAX(idEquipamento) FROM Colheitadeira),
 (SELECT MAX(idPlantio) FROM Plantio), '2024-07-01',
 (SELECT MAX(idArmazem)-4 FROM Armazem),
 (SELECT MAX(idUsuario) FROM Operador)
);

-- Tarefa

    -- TrfPlantio
    INSERT INTO Tarefa
    VALUES (nextval('idTarefa_seq'), '2020-01-01', 6, (SELECT MAX(idUsuario) FROM AdmPlantio), 'Plantio');
      INSERT INTO TrfPlantio VALUES (currval('idTarefa_seq'), (SELECT MAX(idUsuario) FROM Operador));

    INSERT INTO Tarefa
    VALUES (nextval('idTarefa_seq'), '2020-02-01', 5, (SELECT MAX(idUsuario) FROM AdmPlantio), 'Plantio');
      INSERT INTO TrfPlantio VALUES (currval('idTarefa_seq'), (SELECT MAX(idUsuario)-1 FROM Operador));

    -- TrfColheita
    INSERT INTO Tarefa
    VALUES (nextval('idTarefa_seq'), '2020-07-01', 3, (SELECT MAX(idUsuario) FROM AdmPlantio), 'Colheita');
      INSERT INTO TrfColheita VALUES (currval('idTarefa_seq'), (SELECT MAX(idUsuario)-1 FROM Operador));

    INSERT INTO Tarefa
    VALUES (nextval('idTarefa_seq'), '2020-07-01', 1, (SELECT MAX(idUsuario) FROM AdmPlantio), 'Colheita');
      INSERT INTO TrfColheita VALUES (currval('idTarefa_seq'), (SELECT MAX(idUsuario)-1 FROM Operador));

    INSERT INTO Tarefa
    VALUES (nextval('idTarefa_seq'), '2021-01-03', 12, (SELECT MAX(idUsuario) FROM AdmPlantio), 'Colheita');
      INSERT INTO TrfColheita VALUES (currval('idTarefa_seq'), (SELECT MAX(idUsuario) FROM Operador));

-- Trator_Plantio
INSERT INTO Trator_Plantio VALUES (3, 1);
INSERT INTO Trator_Plantio VALUES (4, 1);
INSERT INTO Trator_Plantio VALUES (3, 2);
INSERT INTO Trator_Plantio VALUES (4, 2);
INSERT INTO Trator_Plantio VALUES (3, 3);
INSERT INTO Trator_Plantio VALUES (4, 4);
INSERT INTO Trator_Plantio VALUES (3, 5);
INSERT INTO Trator_Plantio VALUES (4, 5);

-- RecPlantio_espc
INSERT INTO RecPlantio_espc VALUES (1, 1, 10);
INSERT INTO RecPlantio_espc VALUES (1, 2, 15);
INSERT INTO RecPlantio_espc VALUES (2, 1, 51);
INSERT INTO RecPlantio_espc VALUES (2, 2, 20);
INSERT INTO RecPlantio_espc VALUES (2, 3, 15);
INSERT INTO RecPlantio_espc VALUES (3, 2, 80);
INSERT INTO RecPlantio_espc VALUES (3, 3, 100);
INSERT INTO RecPlantio_espc VALUES (3, 4, 40);
INSERT INTO RecPlantio_espc VALUES (4, 1, 80);
INSERT INTO RecPlantio_espc VALUES (4, 3, 100);
INSERT INTO RecPlantio_espc VALUES (4, 4, 40);
INSERT INTO RecPlantio_espc VALUES (5, 1, 100);
INSERT INTO RecPlantio_espc VALUES (5, 2, 120);
INSERT INTO RecPlantio_espc VALUES (5, 3, 150);
INSERT INTO RecPlantio_espc VALUES (5, 4, 400);

-- RecColheita_espc
INSERT INTO RecColheita_espc
VALUES (1, (SELECT MAX(idPlantio) FROM Colheita), (SELECT MAX(idRecColheita)-2 FROM RecColheita), 10);
INSERT INTO RecColheita_espc
VALUES (1, (SELECT MAX(idPlantio) FROM Colheita), (SELECT MAX(idRecColheita)-1 FROM RecColheita), 20);
INSERT INTO RecColheita_espc
VALUES (1, (SELECT MAX(idPlantio) FROM Colheita), (SELECT MAX(idRecColheita) FROM RecColheita), 1);
INSERT INTO RecColheita_espc
VALUES (1, (SELECT MAX(idPlantio)-1 FROM Colheita), (SELECT MAX(idRecColheita)-2 FROM RecColheita), 150);
INSERT INTO RecColheita_espc
VALUES (1, (SELECT MAX(idPlantio)-1 FROM Colheita), (SELECT MAX(idRecColheita)-1 FROM RecColheita), 200);
INSERT INTO RecColheita_espc
VALUES (1, (SELECT MAX(idPlantio)-1 FROM Colheita), (SELECT MAX(idRecColheita) FROM RecColheita), 1000);
INSERT INTO RecColheita_espc
VALUES (1, (SELECT MAX(idPlantio)-2 FROM Colheita), (SELECT MAX(idRecColheita)-2 FROM RecColheita), 120);
INSERT INTO RecColheita_espc
VALUES (1, (SELECT MAX(idPlantio)-2 FROM Colheita), (SELECT MAX(idRecColheita)-1 FROM RecColheita), 250);
INSERT INTO RecColheita_espc
VALUES (1, (SELECT MAX(idPlantio)-2 FROM Colheita), (SELECT MAX(idRecColheita) FROM RecColheita), 1500);
INSERT INTO RecColheita_espc
VALUES (1, (SELECT MAX(idPlantio)-3 FROM Colheita), (SELECT MAX(idRecColheita)-2 FROM RecColheita), 690);
INSERT INTO RecColheita_espc
VALUES (1, (SELECT MAX(idPlantio)-3 FROM Colheita), (SELECT MAX(idRecColheita)-1 FROM RecColheita), 420);
INSERT INTO RecColheita_espc
VALUES (1, (SELECT MAX(idPlantio)-3 FROM Colheita), (SELECT MAX(idRecColheita) FROM RecColheita), 690);
INSERT INTO RecColheita_espc
VALUES (1, (SELECT MAX(idPlantio)-4 FROM Colheita), (SELECT MAX(idRecColheita)-2 FROM RecColheita), 4200);
INSERT INTO RecColheita_espc
VALUES (1, (SELECT MAX(idPlantio)-4 FROM Colheita), (SELECT MAX(idRecColheita)-1 FROM RecColheita), 69420);
INSERT INTO RecColheita_espc
VALUES (1, (SELECT MAX(idPlantio)-4 FROM Colheita), (SELECT MAX(idRecColheita) FROM RecColheita), 20000);

-- Controlador_aloca_RecPlantio
INSERT INTO Controlador_aloca_RecPlantio VALUES ((SELECT idUsuario FROM Controlador), 1);
INSERT INTO Controlador_aloca_RecPlantio VALUES ((SELECT idUsuario FROM Controlador), 2);
INSERT INTO Controlador_aloca_RecPlantio VALUES ((SELECT idUsuario FROM Controlador), 3);
INSERT INTO Controlador_aloca_RecPlantio VALUES ((SELECT idUsuario FROM Controlador), 4);

-- Controlador_aloca_RecColheita
INSERT INTO Controlador_aloca_RecColheita VALUES (1, (SELECT idUsuario FROM Controlador));
INSERT INTO Controlador_aloca_RecColheita VALUES (2, (SELECT idUsuario FROM Controlador));
