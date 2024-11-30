//Prática 1 - Algoritmos e Estruturas de dados 1
//Angelo Pilotto (12542647) e Lucas Miranda (12542838)

#include <iostream>
#include "DataN.cpp"

using namespace std;

int main(void)
{
    bool loop = 1;
    int opt, formatOpt, daysOpt;
    int dia, mes, ano;

    DataN teste1;
    teste1.Cadastra(01, 01, 0001); //Data padrão

    do{
        opt = 100;

        cout << endl << "Teste de cadastramento de datas." << endl;
        cout << "1 - Cadastrar" << endl;
        cout << "2 - Mostrar" << endl;
        cout << "3 - Adicionar dias" << endl;
        cout << "4 - E bissexto?" << endl;
        cout << "0 - Sair" << endl;
        cout << "Digite uma opcao:" << endl;

        fflush(stdin);
        cin >> opt;
        cout << endl;

        switch(opt){
            case 0:
                loop = 0;
                break;
            
            case 1:
                dia, mes, ano = 0;
                cout << "Digite o dia:" << endl;
                cin >> dia;
                cout << "Digite o mes:" << endl;
                cin >> mes;
                cout << "Digite o ano:" << endl;
                cin >> ano;
                teste1.Cadastra(dia, mes, ano);
                break;

            case 2:
                opt = 4;
                cout << "Qual e o formato preferido?" << endl;
                cout << "1 - 01/01/2001" << endl;
                cout << "2 - 01 de janeiro de 2001" << endl;
                cin >> formatOpt;
                cout << endl;
                    switch(formatOpt){
                        case 1:
                            teste1.ImprimeDataFormato1();
                            break;
                        case 2:
                            teste1.ImprimeDataFormato2();
                            break;
                        default:
                            cout << "Opcao invalida!" << endl;
                            break;
                    }
                break;

            case 3:
                cout << "Quantos dias serao adicionados?" << endl;
                cin >> daysOpt;
                if(daysOpt < 8){
                    teste1.AdicionaDias(daysOpt);
                    cout << "Adicionado" << "\n\n";
                }else{
                    cout << "No maximo 7 dias podem ser adicionados" << "\n\n";
                }                
                break;

            case 4:
                if(teste1.Bissexto()){
                    cout << "E ano bissexto." << endl;
                }else{
                    cout << "Nao e ano bissexto." << endl;
                }
                break;
             
            default:
                cout << "Opcao invalida!" << endl;
                break;
        }

        if(loop){
            cout << "Digite enter para voltar ao menu...";
            fflush(stdin);
            cin.ignore();
        }

    }while(loop);

    cout << "Teste concluido.";

}
