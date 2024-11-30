//Prática 1 - Algoritmos e Estruturas de dados 1
//Angelo Pilotto (12542647) e Lucas Miranda (12542838)

#include <iostream>
#include <iomanip>
#include "DataN.h"

using namespace std;

void DataN::Cadastra(int dia, int mes, int ano)
{
    Dia = dia;
    Mes = mes;
    Ano = ano;

    FormatarFevereiro();

    if (Dia > DiasNoMes[Mes] || Dia <= 0)
    {
        cout << "Dia " << Dia << " nao eh valido.\n";
        Dia = 1;
    }
    if (Mes > 12 || Mes <= 0)
    {
        cout << "Mes " << Mes << " nao eh valido.\n";
        Mes = 1;
    }
    if (Ano <= 0)
    {
        cout << "Ano " << Ano << " nao eh valido.\n";
        Ano = 1;
    }
}

void DataN::ImprimeDataFormato1()
{
    cout << setw(2) << setfill('0') << Dia << "/" << setw(2) << setfill('0') << Mes << "/" << Ano << endl;
}

void DataN::ImprimeDataFormato2()
{
    cout << setw(2) << setfill('0') << Dia << " de " << NomeMeses[Mes] << " de " << Ano << endl;
}

void DataN::AdicionaDias(int dias)
{
    if (dias < 0)
    {
        cout << "Deve ser inteiro positivo";
        return;
    }

    Dia += dias;

    if (Dia <= DiasNoMes[Mes])
    {
        return;
    }
    // Passou o mes
    Dia -= DiasNoMes[Mes];
    Mes++;

    if (Mes != 13)
    {
        return;
    }
    // Passou o ano
    Mes = 1;
    Ano++;

    FormatarFevereiro();
}

bool DataN::Bissexto()
{
    if (Ano % 400 == 0)
    {
        return true;
    }
    if (Ano % 100 == 0)
    {
        return false;
    }
    if (Ano % 4 == 0)
    {
        return true;
    }
    return false;
}

void DataN::FormatarFevereiro()
{
    if (Bissexto())
    {
        DiasNoMes[1] = 29;
    }
    else
    {
        DiasNoMes[1] = 28;
    }
}
