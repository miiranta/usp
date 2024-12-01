
-- Equipamento
CREATE SEQUENCE idEquipamento_seq
    AS INTEGER
    INCREMENT 1
    MINVALUE 1
    NO MAXVALUE
    START WITH 1;

    -- Colheitadeira
    -- ?

    -- Trator
    -- ?

-- Usuario
CREATE SEQUENCE idUsuario_seq
    AS INTEGER
    INCREMENT 1
    MINVALUE 1
    NO MAXVALUE
    START WITH 1;

    -- AdmPlantio
    -- ?

    -- Agronomo
    -- ?

    -- Operador
    -- ?

    -- Controlador
    -- ?

    -- AdmSistema
    -- ?

--> Fone_Usuario

-- RecPlantio
CREATE SEQUENCE idRecPlantio_seq
    AS INTEGER
    INCREMENT 1
    MINVALUE 1
    NO MAXVALUE
    START WITH 1;

    -- Combust_plantio
    -- ?

    -- Inseticida
    -- ?

    -- Semente
    -- ?

    -- Fertiliz
    -- ?

-- RecColheita
CREATE SEQUENCE idRecColheita_seq
    AS INTEGER
    INCREMENT 1
    MINVALUE 1
    NO MAXVALUE
    START WITH 1;

    -- Combust_colheita
    -- ?

    -- Saca
    -- ?

-- Armazem
CREATE SEQUENCE idArmazem_seq
    AS INTEGER
    INCREMENT 1
    MINVALUE 1
    NO MAXVALUE
    START WITH 1;

-- Terreno
CREATE SEQUENCE idTerreno_seq
    AS INTEGER
    INCREMENT 1
    MINVALUE 1
    NO MAXVALUE
    START WITH 1;

-- Cultura
CREATE SEQUENCE idCultura_seq
    AS INTEGER
    INCREMENT 1
    MINVALUE 1
    NO MAXVALUE
    START WITH 1;

--> Praga_Cultura

-- Plantio
CREATE SEQUENCE idPlantio_seq
    AS INTEGER
    INCREMENT 1
    MINVALUE 1
    NO MAXVALUE
    START WITH 1;

-- Colheita

-- Tarefa
CREATE SEQUENCE idTarefa_seq
    AS INTEGER
    INCREMENT 1
    MINVALUE 1
    NO MAXVALUE
    START WITH 1;

    -- TrfPlantio
    -- ?

    -- TrfColheita
    -- ?

-- RecPlantio_espc

-- RecColheita_espc

-- Trator_Plantio

-- Controlador_aloca_RecPlantio

-- Controlador_aloca_RecColheita


