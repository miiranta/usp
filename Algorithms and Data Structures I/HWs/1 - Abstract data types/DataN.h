//Prática 1 - Algoritmos e Estruturas de dados 1
//Angelo Pilotto (12542647) e Lucas Miranda (12542838)

#include <iostream>
#include <string>

using namespace std;

class DataN
{
public:
	void Cadastra(int dia, int mes, int ano);
	void ImprimeDataFormato1();
	void ImprimeDataFormato2();
	void AdicionaDias(int dias);
	bool Bissexto();

private:
	int Dia;
	int Mes;
	int Ano;

	const string NomeMeses[13] = {
		"nulo",
		"Janeiro",
		"Fevereiro",
		"Marco",
		"Abril",
		"Maio",
		"Junho",
		"Julho",
		"Agosto",
		"Setembro",
		"Outubro",
		"Novembro",
		"Dezembro"};

	int DiasNoMes[13] = {
		0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

	void FormatarFevereiro();
};
