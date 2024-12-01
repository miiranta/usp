-- -----------------------------------------------------
-- Schema mydb
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Table Equipamento
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Equipamento (
  idEquipamento INTEGER NOT NULL,
  modelo VARCHAR(45) NULL,
  data_compra DATE NULL,
  tipo VARCHAR(45) NULL,
  PRIMARY KEY (idEquipamento))
;


-- -----------------------------------------------------
-- Table Colheitadeira
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Colheitadeira (
  produtividade INTEGER NULL,
  idEquipamento INTEGER NOT NULL,
  PRIMARY KEY (idEquipamento),
  CONSTRAINT fk_Colheitadeira_Equipamento1
    FOREIGN KEY (idEquipamento)
    REFERENCES Equipamento (idEquipamento)
    
    )
;


-- -----------------------------------------------------
-- Table Trator
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Trator (
  potencia INTEGER NULL,
  idEquipamento INTEGER NOT NULL,
  PRIMARY KEY (idEquipamento),
  CONSTRAINT fk_Trator_Equipamento
    FOREIGN KEY (idEquipamento)
    REFERENCES Equipamento (idEquipamento)
    
    )
;


-- -----------------------------------------------------
-- Table Usuario
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Usuario (
  idUsuario INTEGER NOT NULL,
  nome VARCHAR(45) NULL,
  hrs_trab INTEGER NULL,
  senha VARCHAR(45) NULL,
  cargo VARCHAR(45) NOT NULL,
  PRIMARY KEY (idUsuario))
;


-- -----------------------------------------------------
-- Table Controlador
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Controlador (
  formacao VARCHAR(45) NULL,
  desempenho INTEGER NULL,
  idUsuario INTEGER NOT NULL,
  espec VARCHAR(45) NULL,
  PRIMARY KEY (idUsuario),
  CONSTRAINT fk_Controlador_Usuario1
    FOREIGN KEY (idUsuario)
    REFERENCES Usuario (idUsuario)
    
    )
;


-- -----------------------------------------------------
-- Table Armazem
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Armazem (
  idArmazem INTEGER NOT NULL,
  local VARCHAR(45) NULL,
  capacidade INTEGER NULL,
  idControlador INTEGER NOT NULL,
  PRIMARY KEY (idArmazem),
  CONSTRAINT fk_Armazem_Controlador1
    FOREIGN KEY (idControlador)
    REFERENCES Controlador (idUsuario)
    
    )
;


-- -----------------------------------------------------
-- Table RecPlantio
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS RecPlantio (
  idRecPlantio INTEGER NOT NULL,
  nome VARCHAR(45) NULL,
  fornecedor VARCHAR(45) NULL,
  tipo VARCHAR(45) NULL,
  idArmazem INTEGER NOT NULL,
  PRIMARY KEY (idRecPlantio),
  CONSTRAINT fk_RecPlantio_Armazem1
    FOREIGN KEY (idArmazem)
    REFERENCES Armazem (idArmazem)
    
    )
;


-- -----------------------------------------------------
-- Table RecColheita
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS RecColheita (
  idRecColheita INTEGER NOT NULL,
  nome VARCHAR(45) NULL,
  fornecedor VARCHAR(45) NULL,
  tipo VARCHAR(45) NULL,
  idArmazem INTEGER NOT NULL,
  PRIMARY KEY (idRecColheita),
  CONSTRAINT fk_RecColheita_Armazem1
    FOREIGN KEY (idArmazem)
    REFERENCES Armazem (idArmazem)
    
    )
;


-- -----------------------------------------------------
-- Table Combust_plantio
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Combust_plantio (
  tipo VARCHAR(45) NULL,
  idRecPlantio INTEGER NOT NULL,
  PRIMARY KEY (idRecPlantio),
  CONSTRAINT fk_Combust_RecPlantio1
    FOREIGN KEY (idRecPlantio)
    REFERENCES RecPlantio (idRecPlantio)

    )
;


-- -----------------------------------------------------
-- Table Inseticida
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Inseticida (
  combat_praga VARCHAR(45) NULL,
  idRecPlantio INTEGER NOT NULL,
  PRIMARY KEY (idRecPlantio),
  CONSTRAINT fk_Inseticida_RecPlantio1
    FOREIGN KEY (idRecPlantio)
    REFERENCES RecPlantio (idRecPlantio)
    
    )
;


-- -----------------------------------------------------
-- Table Semente
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Semente (
  tipo VARCHAR(45) NULL,
  idRecPlantio INTEGER NOT NULL,
  PRIMARY KEY (idRecPlantio),
  CONSTRAINT fk_Semente_RecPlantio1
    FOREIGN KEY (idRecPlantio)
    REFERENCES RecPlantio (idRecPlantio)
    
    )
;


-- -----------------------------------------------------
-- Table Combust_colheita
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Combust_colheita (
  tipo VARCHAR(45) NULL,
  idRecColheita INTEGER NOT NULL,
  PRIMARY KEY (idRecColheita),
  CONSTRAINT fk_Combust_RecColheita1
    FOREIGN KEY (idRecColheita)
    REFERENCES RecColheita (idRecColheita)
    
    )
;


-- -----------------------------------------------------
-- Table Fertiliz
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Fertiliz (
  N VARCHAR(45) NULL,
  P VARCHAR(45) NULL,
  K VARCHAR(45) NULL,
  idRecPlantio INTEGER NOT NULL,
  PRIMARY KEY (idRecPlantio),
  CONSTRAINT fk_Fertiliz_RecPlantio1
    FOREIGN KEY (idRecPlantio)
    REFERENCES RecPlantio (idRecPlantio)
    
    )
;


-- -----------------------------------------------------
-- Table Saca
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Saca (
  tamanho INTEGER NULL,
  material VARCHAR(45) NULL,
  idRecColheita INTEGER NOT NULL,
  PRIMARY KEY (idRecColheita),
  CONSTRAINT fk_Saca_RecColheita1
    FOREIGN KEY (idRecColheita)
    REFERENCES RecColheita (idRecColheita)
    
    )
;


-- -----------------------------------------------------
-- Table AdmPlantio
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS AdmPlantio (
  formacao VARCHAR(45) NULL,
  idUsuario INTEGER NOT NULL,
  PRIMARY KEY (idUsuario),
  CONSTRAINT fk_AdmPlantio_Usuario1
    FOREIGN KEY (idUsuario)
    REFERENCES Usuario (idUsuario)
    
    )
;


-- -----------------------------------------------------
-- Table Terreno
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Terreno (
  idTerreno INTEGER NOT NULL,
  local VARCHAR(45) NULL,
  descr VARCHAR(45) NULL,
  area  INTEGER NULL,
  N INTEGER NULL,
  P INTEGER NULL,
  K INTEGER NULL,
  idAdmPlantio INTEGER NOT NULL,
  PRIMARY KEY (idTerreno),
  CONSTRAINT fk_Terreno_AdmPlantio1
    FOREIGN KEY (idAdmPlantio)
    REFERENCES AdmPlantio (idUsuario)
    
    )
;


-- -----------------------------------------------------
-- Table Agronomo
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Agronomo (
  formacao VARCHAR(45) NULL,
  terreno_atuac VARCHAR(45) NULL,
  idUsuario INTEGER NOT NULL,
  PRIMARY KEY (idUsuario),
  CONSTRAINT fk_Agronomo_Usuario1
    FOREIGN KEY (idUsuario)
    REFERENCES Usuario (idUsuario)
    
    )
;


-- -----------------------------------------------------
-- Table Cultura
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Cultura (
  idCultura INTEGER NOT NULL,
  nome VARCHAR(45) NULL,
  semente VARCHAR(45) NULL,
  tempo_cultivado INTEGER NULL,
  idAgronomo INTEGER NOT NULL,
  PRIMARY KEY (idCultura),
  CONSTRAINT fk_Cultura_Agronomo1
    FOREIGN KEY (idAgronomo)
    REFERENCES Agronomo (idUsuario)
    
    )
;


-- -----------------------------------------------------
-- Table Plantio
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Plantio (
  idPlantio INTEGER NOT NULL,
  data DATE NULL,
  idCultura INTEGER NOT NULL,
  idTerreno INTEGER NOT NULL,
  PRIMARY KEY (idPlantio),
  CONSTRAINT fk_Plantio_Cultura1
    FOREIGN KEY (idCultura)
    REFERENCES Cultura (idCultura)
    
    ,
  CONSTRAINT fk_Plantio_Terreno1
    FOREIGN KEY (idTerreno)
    REFERENCES Terreno (idTerreno)
    
    )
;


-- -----------------------------------------------------
-- Table Operador
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Operador (
  desempenho INTEGER NULL,
  equipe INTEGER NULL,
  idUsuario INTEGER NOT NULL,
  espec VARCHAR(45) NULL,
  PRIMARY KEY (idUsuario),
  CONSTRAINT fk_Operador_Usuario1
    FOREIGN KEY (idUsuario)
    REFERENCES Usuario (idUsuario)
    
    )
;


-- -----------------------------------------------------
-- Table Colheita
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Colheita (
  producao INTEGER NULL,
  seq_colheita INTEGER NOT NULL,
  idColheitadeira INTEGER NOT NULL,
  idPlantio INTEGER NOT NULL,
  data DATE NULL,
  idArmazem INTEGER NOT NULL,
  idOperador INTEGER NOT NULL,
  PRIMARY KEY (seq_colheita, idPlantio),
  CONSTRAINT fk_Colheita_Colheitadeira1
    FOREIGN KEY (idColheitadeira)
    REFERENCES Colheitadeira (idEquipamento)
    
    ,
  CONSTRAINT fk_Colheita_Plantio1
    FOREIGN KEY (idPlantio)
    REFERENCES Plantio (idPlantio)
    
    ,
  CONSTRAINT fk_Colheita_Armazem1
    FOREIGN KEY (idArmazem)
    REFERENCES Armazem (idArmazem)
    
    ,
  CONSTRAINT fk_Colheita_Operador1
    FOREIGN KEY (idOperador)
    REFERENCES Operador (idUsuario)
    
    )
;


-- -----------------------------------------------------
-- Table Tarefa
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Tarefa (
  idTarefa INTEGER NOT NULL,
  dta_emissao DATE NULL,
  prazo INTEGER NULL,
  idAdmPlantio INTEGER NOT NULL,
  tipo VARCHAR(45) NULL,
  PRIMARY KEY (idTarefa),
  CONSTRAINT fk_Tarefa_AdmPlantio1
    FOREIGN KEY (idAdmPlantio)
    REFERENCES AdmPlantio (idUsuario)
    
    )
;


-- -----------------------------------------------------
-- Table TrfPlantio
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS TrfPlantio (
  idTarefa INTEGER NOT NULL,
  idOperador INTEGER NOT NULL,
  PRIMARY KEY (idTarefa),
  CONSTRAINT fk_TrfPlantio_Tarefa1
    FOREIGN KEY (idTarefa)
    REFERENCES Tarefa (idTarefa)
    
    ,
  CONSTRAINT fk_TrfPlantio_Operador1
    FOREIGN KEY (idOperador)
    REFERENCES Operador (idUsuario)
    
    )
;


-- -----------------------------------------------------
-- Table TrfColheita
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS TrfColheita (
  idTarefa INTEGER NOT NULL,
  idOperador INTEGER NOT NULL,
  PRIMARY KEY (idTarefa),
  CONSTRAINT fk_TrfColheita_Tarefa1
    FOREIGN KEY (idTarefa)
    REFERENCES Tarefa (idTarefa)
    
    ,
  CONSTRAINT fk_TrfColheita_Operador1
    FOREIGN KEY (idOperador)
    REFERENCES Operador (idUsuario)
    
    )
;


-- -----------------------------------------------------
-- Table AdmSistema
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS AdmSistema (
  nv_acesso INTEGER NULL,
  idUsuario INTEGER NOT NULL,
  PRIMARY KEY (idUsuario),
  CONSTRAINT fk_AdmSistema_Usuario1
    FOREIGN KEY (idUsuario)
    REFERENCES Usuario (idUsuario)
    
    )
;


-- -----------------------------------------------------
-- Table Praga_Cultura
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Praga_Cultura (
  Praga VARCHAR(45) NOT NULL,
  idCultura INTEGER NOT NULL,
  PRIMARY KEY (Praga, idCultura),
  CONSTRAINT fk_Praga_Cultura1
    FOREIGN KEY (idCultura)
    REFERENCES Cultura (idCultura)
    
    )
;


-- -----------------------------------------------------
-- Table Fone_Usuario
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Fone_Usuario (
  Fone CHAR(11) NOT NULL,
  idUsuario INTEGER NOT NULL,
  PRIMARY KEY (Fone, idUsuario),
  CONSTRAINT fk_Fone_Usuario_Usuario1
    FOREIGN KEY (idUsuario)
    REFERENCES Usuario (idUsuario)
    
    )
;


-- -----------------------------------------------------
-- Table Trator_Plantio
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Trator_Plantio (
  idTrator INTEGER NOT NULL,
  idPlantio INTEGER NOT NULL,
  PRIMARY KEY (idTrator, idPlantio),
  CONSTRAINT fk_Trator_has_Plantio_Trator1
    FOREIGN KEY (idTrator)
    REFERENCES Trator (idEquipamento)
    
    ,
  CONSTRAINT fk_Trator_has_Plantio_Plantio1
    FOREIGN KEY (idPlantio)
    REFERENCES Plantio (idPlantio)
    
    )
;


-- -----------------------------------------------------
-- Table RecPlantio_espc
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS RecPlantio_espc (
  idPlantio INTEGER NOT NULL,
  idRecPlantio INTEGER NOT NULL,
  qtd INTEGER NULL,
  PRIMARY KEY (idPlantio, idRecPlantio),
  CONSTRAINT fk_Plantio_has_RecPlantio_Plantio1
    FOREIGN KEY (idPlantio)
    REFERENCES Plantio (idPlantio)
    
    ,
  CONSTRAINT fk_Plantio_has_RecPlantio_RecPlantio1
    FOREIGN KEY (idRecPlantio)
    REFERENCES RecPlantio (idRecPlantio)
    
    )
;


-- -----------------------------------------------------
-- Table RecColheita_espc
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS RecColheita_espc (
  seq_colheita INTEGER NOT NULL,
  idPlantio INTEGER NOT NULL,
  idRecColheita INTEGER NOT NULL,
  qtd INTEGER NULL,
  PRIMARY KEY (seq_colheita, idPlantio, idRecColheita),
  CONSTRAINT fk_Colheita_has_RecColheita_Colheita1
    FOREIGN KEY (seq_colheita , idPlantio)
    REFERENCES Colheita (seq_colheita , idPlantio)
    
    ,
  CONSTRAINT fk_Colheita_has_RecColheita_RecColheita1
    FOREIGN KEY (idRecColheita)
    REFERENCES RecColheita (idRecColheita)
    
    )
;


-- -----------------------------------------------------
-- Table Controlador_aloca_RecPlantio
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Controlador_aloca_RecPlantio (
  idControlador INTEGER NOT NULL,
  idRecPlantio INTEGER NOT NULL,
  PRIMARY KEY (idControlador, idRecPlantio),
  CONSTRAINT fk_Controlador_has_RecPlantio_Controlador1
    FOREIGN KEY (idControlador)
    REFERENCES Controlador (idUsuario)
    
    ,
  CONSTRAINT fk_Controlador_has_RecPlantio_RecPlantio1
    FOREIGN KEY (idRecPlantio)
    REFERENCES RecPlantio (idRecPlantio)
    
    )
;


-- -----------------------------------------------------
-- Table Controlador_aloca_RecColheita
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS Controlador_aloca_RecColheita (
  idRecColheita INTEGER NOT NULL,
  idControlador INTEGER NOT NULL,
  PRIMARY KEY (idRecColheita, idControlador),
  CONSTRAINT fk_RecColheita_has_Controlador_RecColheita1
    FOREIGN KEY (idRecColheita)
    REFERENCES RecColheita (idRecColheita)
    
    ,
  CONSTRAINT fk_RecColheita_has_Controlador_Controlador1
    FOREIGN KEY (idControlador)
    REFERENCES Controlador (idUsuario)
    
    )
;


