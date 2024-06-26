// Alumno: Álvaro Rodríguez Gallardo
// DNI: 77034155W
// Correo: alvaro155w@correo.ugr.es
// Grupo 2 (Miércoles, 17.30-19.30)
#include <iostream>
#include <vector>
#include "practica2.hpp"

using namespace std;

// ******************************************************************************************************************************************************
// *************************************************** FUNCIÓN PRINCIPAL ********************************************************************************
// ******************************************************************************************************************************************************

int main(int argc, char **argv){
    long int semilla;

    if(argc <= 1){
        cout << "Introduzca una semilla"<<endl;
        return -1;
    } else {
        semilla = atoi(argv[1]);
        Random::seed(semilla);
        cout <<"Semilla usada: " <<semilla <<endl;
    
        resultadosP1("original");
        resultadosP1("BL");
        resultadosP1("RELIEF");
        resultadosP2("AGG-BX");
        resultadosP2("AGG-CA");
        resultadosP2("AGE-BX");
        resultadosP2("AGE-CA");
        resultadosP2("AM-1");
        resultadosP2("AM-01");
        resultadosP2("AM-mej");
    }

    return 0;
}

