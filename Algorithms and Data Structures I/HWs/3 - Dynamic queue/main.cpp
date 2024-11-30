//Prática 3 - Algoritmos e Estruturas de dados 1
//Lucas Miranda (12542838)

#include <iostream>
#include <iomanip>
#include <string>
#include "queue.cpp"
using namespace std;

const int NdiasDaSemana = 5;
const int Nhorarios     = 2;
const int Nvagas        = 100;
const string diasDaSemana[NdiasDaSemana]    = {"segunda", "terca", "quarta", "quinta", "sexta"};
const string horarios[Nhorarios]            = {"manha", "tarde"};
Queue fila[NdiasDaSemana][Nhorarios];

void mostrarCalendario();
void mostrarSlot(int dia, int horario);
void vacinarPessoa(int dia, int horario);
void adicionarPessoa(int dia, int horario, pessoa p);

int main(){

    bool loop = true;
    int op, opDia, opHorario;
    pessoa p;

    do{
        op = 5;

        cout << endl << "Digite o numero correspondente a opcao desejada: " << endl;
        cout << "1 - Adicionar pessoa" << endl;
        cout << "2 - Vacinar pessoa" << endl;
        cout << "3 - Mostrar calendario completo" << endl;
        cout << "4 - Mostrar slot" << endl;
        cout << "0 - Sair" << endl;

        cin >> op;

        switch(op){
            case 1:
                cout << "Digite o nome: ";
                fflush(stdin);
                getline(cin, p.nome);
                
                cout << "Digite o endereco: ";
                fflush(stdin);
                getline(cin, p.endereco);
                
                cout << "Digite o cpf (sem pontos ou hifen): ";
                fflush(stdin);
                getline(cin, p.cpf);

                cout << "Digite a idade: ";
                fflush(stdin);
                cin >> p.idade;

                cout << "1 - Segunda " << endl;
                cout << "2 - Terca " << endl;
                cout << "3 - Quarta " << endl;
                cout << "4 - Quinta " << endl;
                cout << "5 - Sexta " << endl;
                cout << "Digite o dia da semana: ";
                fflush(stdin);
                cin >> opDia;

                cout << "1 - Manha " << endl;
                cout << "2 - Tarde " << endl;
                cout << "Digite o horario: ";
                fflush(stdin);
                cin >> opHorario;

                cout << endl;
                adicionarPessoa(opDia-1, opHorario-1, p);
                break;

            case 2:
                cout << "1 - Segunda " << endl;
                cout << "2 - Terca " << endl;
                cout << "3 - Quarta " << endl;
                cout << "4 - Quinta " << endl;
                cout << "5 - Sexta " << endl;
                cout << "Digite o dia da semana: ";
                fflush(stdin);
                cin >> opDia;

                cout << "1 - Manha " << endl;
                cout << "2 - Tarde " << endl;
                cout << "Digite o horario: ";
                fflush(stdin);
                cin >> opHorario;
                
                cout << endl;
                vacinarPessoa(opDia-1, opHorario-1);
                break;

            case 3:
                cout << endl;
                mostrarCalendario();
                break;

            case 4:
                cout << "1 - Segunda " << endl;
                cout << "2 - Terca " << endl;
                cout << "3 - Quarta " << endl;
                cout << "4 - Quinta " << endl;
                cout << "5 - Sexta " << endl;
                cout << "Digite o dia da semana: ";
                fflush(stdin);
                cin >> opDia;

                cout << "1 - Manha " << endl;
                cout << "2 - Tarde " << endl;
                cout << "Digite o horario: ";
                fflush(stdin);
                cin >> opHorario;

                cout << endl;
                mostrarSlot(opDia-1, opHorario-1);
                break;

            case 0:
                cout << endl;
                loop = false;
                break;

            default:
                break;
        }

        if(loop){
            cout << "Digite enter para voltar ao menu...";
            fflush(stdin);
            cin.ignore();
        }

    }while(loop);

    cout << "Saindo...";
}

void mostrarCalendario(){
    for(int i=0; i<5; i++){
        for(int j=0; j<2; j++){
            cout << ">";
            cout << setw(8) << diasDaSemana[i] ;
            cout << setw(1) << " " << horarios[j] << ": ";

            if(fila[i][j].empty()){
                cout << "  " << "Vazio." << endl;
                continue;
            }

            int tamanho = fila[i][j].size();
            cout << "  Ha " << tamanho << " pessoas registradas. " << Nvagas - tamanho << " vagas." << endl;
        }
    }
}

void mostrarSlot(int dia, int horario){
    //Horario valido?
    if(dia > NdiasDaSemana-1 || horario > Nhorarios-1 || dia < 0 || horario < 0){
        cout << "Dia/Horario invalido." << endl;
        return;
    }
    
    fila[dia][horario].print();
}

void vacinarPessoa(int dia, int horario){
    //Horario valido?
    if(dia > NdiasDaSemana-1 || horario > Nhorarios-1 || dia < 0 || horario < 0){
        cout << "Dia/Horario invalido." << endl;
        return;
    }
    
    if(fila[dia][horario].empty()){
        cout << "Fila vazia." << endl;
        return;
    }

    pessoa p;
    fila[dia][horario].serve(p);
    cout << p.nome << " foi vacinado e retirado da fila." << endl;
}

void adicionarPessoa(int dia, int horario, pessoa p){
    //Horario valido?
    if(dia > NdiasDaSemana-1 || horario > Nhorarios-1 || dia < 0 || horario < 0){
        cout << "Dia/Horario invalido." << endl;
        return;
    }
    
    //Slot cheio?
    if(fila[dia][horario].size()>=Nvagas){
        cout << "Fila cheia." << endl;
        return;
    }

    //Nome muito grande?
    if(p.nome.size()>200){
        cout << "Nome ultrapassou 200 chars." << endl;
        return;
    }

    //Endereco muito grande?
    if(p.endereco.size()>400){
        cout << "Endereco ultrapassou 400 chars." << endl;
        return;
    }

    //Idade abaixo de 5?
    if(p.idade<5){
        cout << "Idade abaixo de 5 anos." << endl;
        return;
    }

    fila[dia][horario].append(p);
    cout << p.nome << " foi adicionada na fila." << endl;
}
