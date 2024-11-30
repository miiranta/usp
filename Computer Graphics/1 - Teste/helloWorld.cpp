//*****************************************************
//
// pratica1.cpp
// Um programa OpenGL simples que abre uma janela GLUT
// e desenha duas linhas verdes cruzadas
// 
//*****************************************************
#include <windows.h>
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>


#include <iostream>

// função para fazer o desenho
void Desenha(void)
{
    // limpa a janela de visualização com a cor de fundo especificada
    glClear(GL_COLOR_BUFFER_BIT);
    // define a cor da linha
    glColor3f(0.0, 1.0, 0.0);
    // desenha linhas na tela
    glBegin(GL_LINES);
        // define os vértices das linhas
        // linha 1
        glVertex2f(100.0, 100.0);
        glVertex2f(400.0, 400.0);
        // linha 2
        glVertex2f(100.0, 400.0);
        glVertex2f(400.0, 100.0);
    glEnd();

    // executa os comandos opengl
    glFlush();
}

// inicializa parametros de rendering
void Inicializa(void)
{
    // cor de fundo da janela de visualizaçao
    glClearColor(1.0, 1.0, 1.0, 1.0);
    // espessura da linha
    glLineWidth(3.0);
    // tamanho da janela de visualização
    gluOrtho2D(0.0, 500.0, 0.0, 500.0);
    
}

int main(int argc, char** argv)
{
      std::cout << "Pratica 1\n";

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);  // usa apenas um buffer e o sistema rgb
    glutCreateWindow("Aula Pratica 1"); // nome da janela
    glutDisplayFunc(Desenha);  // funçao com os dados primitivos
    Inicializa(); // inicialização da tela e do fundo
    glutMainLoop(); // executa todo o processo
}
