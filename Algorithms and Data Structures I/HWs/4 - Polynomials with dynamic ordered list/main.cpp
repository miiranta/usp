//Prática 4 - Algoritmos e Estruturas de dados 1
//Lucas Miranda (12542838)

#include <iostream>
#include "list.cpp"

using namespace std;

int main(){
    OrderedList p, q, result;
    exponentType exp;
    coeficientType coef;

    bool loop = true;
    int op;
    do{
        op = 6;

        cout << endl << "P(x)" << endl;
        p.print();

        cout << endl << "Q(x)" << endl;
        q.print();

        cout << endl << "Digite o numero correspondente a opcao desejada: " << endl;
        cout << "1 - Modificar polinomio P(x)" << endl;
        cout << "2 - Modificar polinomio Q(x)" << endl;
        cout << "3 - Retornar o grau de P(x) e Q(x)" << endl;
        cout << "4 - Imprimir P(x) + Q(x)" << endl;
        cout << "5 - Imprimir P(x) * Q(x)" << endl;
        cout << "0 - Sair" << endl;

        cin >> op;

        switch(op){
            case 1:
                cout << "Digite o expoente: " << endl;
                cin >> exp;
                cout << "Digite o coeficiente: " << endl;
                cin >> coef;
                p.replace(exp, coef);
                break;
            case 2:
                cout << "Digite o expoente: " << endl;
                cin >> exp;
                cout << "Digite o coeficiente: " << endl;
                cin >> coef;
                q.replace(exp, coef);
                break;
            case 3:
                cout << "O grau de P(x) e " << p.degree() << endl;
                cout << "O grau de Q(x) e " << q.degree() << endl;
                break;
            case 4:
                result.overwriteByListSum(p, q);
                cout << "P(x) + Q(x) = ";
                result.print();
                break;
            case 5:
                result.overwriteByListProduct(p, q);
                cout << "P(x) * Q(x) = ";
                result.print();
                break;
            case 0:
                loop = false;
                break;
            default:
                cout << "Opcao invalida." << endl;
                break;
        }

        if(loop){
            cout << "Digite enter para voltar ao menu...";
            fflush(stdin);
            cin.ignore();
        }

    }while(loop);
    
    cout << "Saindo..." << endl;
}
