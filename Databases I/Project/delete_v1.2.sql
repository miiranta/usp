
DELETE FROM Armazem;
DELETE FROM Colheita;
DELETE FROM Fone_Usuario;

DELETE FROM Trator;
DELETE FROM Colheitadeira;

DELETE FROM Equipamento;

DELETE FROM Trator_Plantio;
DELETE FROM RecPlantio_espc;
DELETE FROM RecColheita_espc;
DELETE FROM Controlador_aloca_RecPlantio;
DELETE FROM Controlador_aloca_RecColheita;

DELETE FROM Praga_Cultura;


DELETE FROM TrfPlantio;
DELETE FROM TrfColheita;

DELETE FROM Tarefa;

DELETE FROM Plantio;

DELETE FROM Cultura;

DELETE FROM Terreno;

DELETE FROM AdmPlantio;
DELETE FROM AdmSistema;
DELETE FROM Agronomo;
DELETE FROM Controlador;
DELETE FROM Operador;

DELETE FROM Usuario;

DELETE FROM Combust_plantio;
DELETE FROM Inseticida;
DELETE FROM Semente;
DELETE FROM Fertiliz;

DELETE FROM RecPlantio;

DELETE FROM Combust_colheita;
DELETE FROM Saca;

DELETE FROM RecColheita;

ALTER SEQUENCE idEquipamento_seq RESTART;
ALTER SEQUENCE idUsuario_seq RESTART;
ALTER SEQUENCE idRecPlantio_seq RESTART;
ALTER SEQUENCE idRecColheita_seq RESTART;
ALTER SEQUENCE idArmazem_seq RESTART;
ALTER SEQUENCE idTerreno_seq RESTART;
ALTER SEQUENCE idCultura_seq RESTART;
ALTER SEQUENCE idPlantio_seq RESTART;
ALTER SEQUENCE idTarefa_seq RESTART;


