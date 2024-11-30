//Atividade Pratica 1
//Nome:       Lucas Miranda
//Numero USP: 12542838
//Professor:  Renato Tinós

#include <stdio.h>
#include <string.h>
#define MAXSIZE 1000

void preDefinidos();
int mostrarMenu();
void mostrarAluno();
void adicionarAluno();
float calculaMedia(float prova1, float prova2);
void sleeper();

//Setup do struct
struct aluno {
    char nome[MAXSIZE];
    int nUSP;
    float notaProva1;
    float notaProva2;
    float media;
};

//Vars
int alunoQuantidade = 0;
struct aluno alunoDados[MAXSIZE];

void main(){
    int loop = 1;

    //Define 3 alunos exemplo (nUSP = 12542838, 2, 3)
    preDefinidos();

    //Cria um loop até a opção 0 ser digitada em mostrarMenu()
    do{
        loop = mostrarMenu();
    }while(loop == 1);
}

void preDefinidos(){
//3 EXEMPLOS PRÉ-DEFINIDOS--------------

    //nUSP = 12542838
    strcpy(alunoDados[1].nome, "Lucas Miranda\n");
    alunoDados[1].nUSP       = 12542838;
    alunoDados[1].notaProva1 = 10;
    alunoDados[1].notaProva2 = 10;
    alunoDados[1].media      = calculaMedia(alunoDados[1].notaProva1, alunoDados[1].notaProva2);
    alunoQuantidade++;

    //nUSP = 2
    strcpy(alunoDados[2].nome, "Joaozinho\n");
    alunoDados[2].nUSP       = 2;
    alunoDados[2].notaProva1 = 3;
    alunoDados[2].notaProva2 = 7;
    alunoDados[2].media      = calculaMedia(alunoDados[2].notaProva1, alunoDados[2].notaProva2);
    alunoQuantidade++;

    //nUSP = 3
    strcpy(alunoDados[3].nome, "Pedrinho\n");
    alunoDados[3].nUSP       = 3;
    alunoDados[3].notaProva1 = 0;
    alunoDados[3].notaProva2 = 7;
    alunoDados[3].media      = calculaMedia(alunoDados[3].notaProva1, alunoDados[3].notaProva2);
    alunoQuantidade++;

//------------------------------------
}

int mostrarMenu(){
    int loop = 1, option = 0;

    //Menu
    printf("\n================================\n\n");
    printf("Digite o numero correspondente + enter:\n");
    printf("1 - Adicionar Aluno\n");
    printf("2 - Mostrar Aluno\n");
    printf("0 - Sair\n");
    printf("\n================================\n\n");

    //Pega a opção escolhida
    fflush(stdin);
    scanf("%d", &option);

    if(option == 0){loop = 0;}
    if(option == 1){adicionarAluno();}
    if(option == 2){mostrarAluno();}

    return loop;
}

void mostrarAluno(){
    int nUSP = 0, i = 0, j = 0;

    printf("\n================================\n\n");
    printf("Digite o numero USP:\n ");

    //Pega o nUSP
    fflush(stdin);
    scanf("%d", &nUSP);

    //Confere se é válido
    if(nUSP == 0){
        printf("\nNumero USP nao pode ser zero nem chars!\n");
        printf("\n================================\n\n");
        sleeper();
        return;
    }

    //Testa para ver se existe
    for(i = 0; i<=alunoQuantidade; i++){
        if(alunoDados[i].nUSP == nUSP){
            //SE EXISTE
            printf("\nAluno encontrado:\n\n");
            printf("Nome - ");
            for(j = 0; alunoDados[i].nome[j]; j++){
                printf("%c", alunoDados[i].nome[j]);
            }
            printf("nUSP - %d\n", alunoDados[i].nUSP);
            printf("Prova 1 - %f\n", alunoDados[i].notaProva1);
            printf("Prova 2 - %f\n", alunoDados[i].notaProva2);
            printf("Media - %f\n", alunoDados[i].media);
            printf("\n================================\n\n");
            sleeper();
            return;
        }
    }

    //SE NÃO EXISTE
    printf("\nAluno nao encontrado!\n");
    printf("\n================================\n\n");
    sleeper();
    return;
}

void adicionarAluno(){
    int nUSP = 0, i = 0;
    char nome[MAXSIZE];
    float prova1 = 0, prova2 = 0;

    printf("\n================================\n\n");
    printf("Digite o numero USP:\n ");

    //Pega o nUSP
    fflush(stdin);
    scanf("%d", &nUSP);

    //Verifica se é válido
    if(nUSP == 0){
        printf("\nNumero USP nao pode ser zero nem chars!\n");
        printf("\n================================\n\n");
        sleeper();
        return;
    }

    //Testa para ver se já está registrado
    for(i = 0; i<=alunoQuantidade; i++){
        if(alunoDados[i].nUSP == nUSP){
            //SE está registrado
            printf("\nAluno ja registrado!\n");
            printf("\n================================\n\n");
            sleeper();
            return;
        }
    }

    //SE NÃO está registrado
    alunoQuantidade++;

    //Define os dados no struct
    alunoDados[alunoQuantidade].nUSP = nUSP;

    printf("Digite o nome:\n ");
    fflush(stdin);
    fgets(nome, MAXSIZE, stdin);
    strcpy(alunoDados[alunoQuantidade].nome, nome);

    printf("Digite a nota da prova 1:\n ");
    fflush(stdin);
    scanf("%f", &prova1);
    alunoDados[alunoQuantidade].notaProva1 = prova1;

    printf("Digite a nota da prova 2:\n ");
    fflush(stdin);
    scanf("%f", &prova2);
    alunoDados[alunoQuantidade].notaProva2 = prova2;

    alunoDados[alunoQuantidade].media = calculaMedia(prova1, prova2);

    printf("\n================================\n\n");
    return;
}

float calculaMedia(float prova1, float prova2){
    return (prova1+prova2)/2;
}

void sleeper(){
    int dummy;
    printf("Pressione 0 + enter para voltar ao menu...\n ");
    fflush(stdin);
    scanf("%d", &dummy);
}