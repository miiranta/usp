-- OBSs
-- 1. Da pra sobrecarregar as funções
-- 2. Nome dos parametros repassados: $1, $2...
-- 3. Case INsensitive
-- 4. Tabelas já são variaveis reconhecidas
-- 5. Pode-se usar qualquer comando do sql, SUM(), AVG(), DISTINCT

-- INSERT / UPDATE / DELETE / SELECT !
    INSERT INTO Tabela (col, col...) VALUES (val, val...);
    UPDATE Tabela SET campo = "",...
    DELETE FROM Tabela ...
    SELECT campo,... FROM Tabela
    --[WHERE] em tudo!

-- Função Básica

    CREATE [OR REPLACE] FUNCTION nome ([tipo_param [, tipo_param, …]])
    RETURNS tipo_retorno
    AS
    $$
        DECLARE
            variável tipo_variável;
            …
        BEGIN
            instrução;
            …
            RETURN valor_retorno;
        END;
    $$
    LANGUAGE ‘plpgsql’;


-- Atribuição

    DECLARE
        result INT

    BEGIN
        result := ($1 + $2);
        -- "=" funciona também
        -- Se a expressão não for do mesmo tipo, uma conversão será tentada.

        RETURN result;
    END;


-- Exec func
    SELECT func_nome(params...);


-- Deletar func
    DROP FUNCTION func_nome(todos_os_params_sem_falta!...)


-- Commentarios
/* */


-- Apelidos 

    DECLARE
        result INT
        hello ALIAS FOR $1
        world ALIAS FOR $2

    --OU
    CREATE [OR REPLACE] FUNCTION nome (hello NUMERIC, world NUMERIC)


-- Declaração de vars

    DECLARE
        oi NUMERIC := 1000; --OK!
        tu NUMERIC := 1000 * 2 * oi; --NÃO FUNCIONA!!
        do CONSTANT NUMERIC := 1000; --OK!


-- Tipos

    DECLARE
        a NUMERIC
        b REAL

        x Tabela.Campo%TYPE; --Tipo de um campo
        y Tabela%ROWTYPE; --Tipo de uma tabela
        z RECORD; --Coringa


-- Armazenando consulta

    SELECT INTO var_que_armazena campos_da_tabela FROM Tabela WHERE ""="";
    --Campos_da_tabela pode ser *


-- Msg

    RAISE NOTICE 'Oi oi %', var;

    RAISE EXCEPTION 'Para o bloco!';


-- Condicionais

    IF "<>=" "OR AND" "IS NULL"
    THEN
        inst;
    ELSE
        inst;
    END IF;


-- Repetição

    -- Loop
    LOOP
        ...;
        EXIT WHEN cond; 
        --OU
        IF COND THEN EXIT END IF;

    END LOOP;

    -- While
    WHILE cond LOOP

    END LOOP;

    --For-in
    FOR i IN valor_inicio ... valor_fim LOOP

    END LOOP;

    --For-in-Select
    FOR i IN SELECT ... LOOP --Uma vez para cada tupla :)

    END LOOP;


-- Retornando lista
    
    RETURN QUERY SELECT cod_func FROM Func;


-- Retornando seleção

    RETURN QUERY SELECT nome, salario/2 FROM Func
    WHERE nro_depto = p_nro_depto;


-- Perform e Execute
    -- Serve para não precisar guardar o resultado

    --Execute
    -- Quando a consulta é montada na hora
    CREATE OR REPLACE FUNCTION conta_tempo (VARCHAR)
    ...
    EXECUTE $1;

    --Perform
    -- Quando se tem a consulta de antemão
    -- Sem exemplo ;(


-- Cursor
    -- Uma estrutura mais flexível para ler o BD

    DECLARE 
        nome_cursor CURSOR FOR SELECT * FROM Tabela;

    --É necessário abrí-lo
    BEGIN
        OPEN nome_cursor;
        ...
        CLOSE nome_cursor;

    --FETCH para recuperar dados para uma variavel
    BEGIN
        ...
        FETCH nome_cursor INTO var_rowtype_ou_record; --Recupera uma tupla por vez

    --FOUND é uma var bool q retorna true se o fetch achou alguma coisa
    FETCH cs_clientes INTO um_cliente;
    WHILE FOUND LOOP
        i := i + 1;
        RAISE NOTICE ‘Nome do %o cliente: %’, i, um_cliente.nome;
        FETCH cs_clientes INTO um_cliente;
    END LOOP;


-- Cursor com parametros

    DECLARE 
        nome_cursor CURSOR (a tipo, b tipo...) FOR
        SELECT * FROM Cliente WHERE cidade = pc_cidade;

    --Abrindo
    BEGIN
        OPEN nome_cursor(a, b)
        ...
        CLOSE nome_cursor;


-- Trigger
    -- Exec uma função especial quando há um INSERT, DELETE, UPDATE

    -- Criando
    CREATE TRIGGER nome_gatilho [BEFORE | AFTER] [INSERT | DELETE | UPDATE (campo) [OR …]]
    ON nome_tabela [FOR EACH ROW]
    EXECUTE FUNCTION nome_função()

    CREATE FUNCTION nome_função() RETURNS TRIGGER
    AS
    $$ 
        BEGIN
        ...
        END;
    $$ LANGUAGE 'plpgsql';

    -- OLD => Linha antiga (excluida ou alterada)
    -- NEW => Linha nova (inserida ou alterada)

    -- Excluindo
    DROP TRIGGER nome_gatilho ON nome_tabela;
