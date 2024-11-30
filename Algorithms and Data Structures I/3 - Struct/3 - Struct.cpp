#include <iostream>
using namespace std;

//No need for using typedef!
struct alunos{
    string nome;
    int idade;
};
alunos listaAlunos[20];

//C way...
//typedef struct {
//    string nome;
//    int idade;
//} alunos;
//alunos listaAlunos[20];

//You can also put a struct variable inside another struct!
struct inside {
    int evenMoreInside;
};
struct outside {
    inside in;
};


int main(){

    listaAlunos[4].nome = "Lucas";
    cout << listaAlunos[4].nome;

}

