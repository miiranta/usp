#include <GL/glut.h>
#include <iostream>
#include <math.h>

//Consts
#define FPS 60

//Prototypes
void Inicializa();
void Desenha();
void Timer(int v);
void AlteraTamanhoJanela(GLint largura, GLint altura);
void Camera();
void Movimento();
void Gamemode();
void RecriarMapa();
void Hitbox();
void Animacao();
void ApontarBloco();
void GerenciaMouse(int button, int state, int x, int y);
void GerenciaMouseNav(int x, int y);
void GerenciaTecladoDown(unsigned char key, int x, int y);
void GerenciaTecladoUp(unsigned char key, int x, int y);
void GerenciaEspecialDown(int key, int x, int y);
void GerenciaEspecialUp(int key, int x, int y);

void DesenhaCubo(int x, int y, int z, int tipo);
void DesenhaHitbox();
void DesenhaBlocoApontado();
void DesenhaItens();
void DesenhaNuvens();
void DesenhaMapa();
void GerarMapa();
void LimparMapa();
void AdicionarItem(int x, int y, int z, int tipo);
void DistribuirNuvens();
void LimparNuvens();
void CriarPilastra(int x, int y, int z);
void CriarOceano(int z, int startx, int endx, int starty, int endy);
void CriarArvore(int x, int y, int z);
float CalculaPonto(char c, double t);
void CalculaCurvas();

//Config
unsigned int MAP_SEED = 11425;
float MAP_LENGTH = 32; //63 is max

float MAX_WALK_SPEED = 2;
float WALK_ACCELERATION = 0.2;

float MAX_CAMERA_SPEED = 2;
float CAMERA_ACCELERATION = 0.2;

bool COLLISION = true;

bool SHOW_CAMERA_HITBOX = false;
float CAMERA_HITBOX_SIZE = 15; //< Block size

bool SHOW_BEZIER_CURVES = false;

bool GRAVITY = false;
float MAX_GRAVITY_SPEED = 20;
float GRAVITY_ACCELERATION = 0.6;

bool SHOW_RIGHT_CLICK = true;

float BLOCK_SIZE = 50;
float ITEM_SIZE = 10;

float JUMP_SPEED = 10;

float SPAWN_COORDS[3] = {0, 0, 40};

float CAM_MODE = 1; //0 - Minecraft Cam | 1 - Free Cam

float GAME_MODE = 1; //0 - Survival | 1 - Spectator

bool HIDE_CURSOR = true;
bool ENABLE_MOUSE_NAV = true;
float MOUSE_SENSE = 0.2;

//Global vars
int largura = 1200;
int altura = 800;

GLfloat fAspect, camAngle;

bool keystates[256] = {false};
bool specialstates[256] = {false};
int w = 0, a = 0, s = 0, d = 0;
float mousex = 0, mousey = 0;

int mapa[64][64][64] = {0};

float cam[3] = {SPAWN_COORDS[0] * BLOCK_SIZE, SPAWN_COORDS[1] * BLOCK_SIZE, SPAWN_COORDS[2] * BLOCK_SIZE};

float normal[3] = {1,1,0}; 
float normal90[3] = {0,0,0};
float up[3] = {0,0,1};

float walk_speed = 0;
float camera_speed = 0;
float gravity_speed = 0;

float current_camera_hitbox[8][3] = {0};

int bloco_apontado[3] = {-1};
int bloco_apontado_anterior[3] = {-1};
int bloco_selecionado = 1;

float nuvens[100][3] = {0};
float nuvens_speed = 0.01;
float nuvens_rate = 0.0001;
int nuvem_altura = 49;
int delta_altura = 60;
int nuvens_limite = floor(MAP_LENGTH/3);

float itens[100][5] = {0}; //X Y Z TIPO ANIMAÇÃO
int itens_limite = 100;

float Bx[4] = {0}, By[4] = {0}, Bz[4] = {0};
int itens_quantidade = 0;

bool debug = false;

float luzAmbiente[4]={0.3, 0.3, 0.3, 1.0};   // {R, G, B, alfa}
float luzDifusa[4]={1.0, 1.0, 1.0, 0.2};	   // o 4o componente, alfa, controla a opacidade/transparência da luz
float luzEspecular[4]={1.0, 1.0, 1.0, 1.0};
float posicaoLuz[4]={0 * BLOCK_SIZE, 0 * (BLOCK_SIZE), 65 * (BLOCK_SIZE), 1.0}; // {x, y, z, 1.0} = luz posicional 
	
float especularidade[4]={0.1, 0.1, 0.1, 1.0}; // Capacidade de brilho do material
int especMaterial = 100;

//Basic functions
int main(int argc, char** argv){
    glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowPosition(200,200);
    
    fAspect = (GLfloat)largura / (GLfloat)altura;
    glutInitWindowSize(largura,altura);
    glutCreateWindow("Minecraft 2"); 

	glutDisplayFunc(Desenha);
	glutReshapeFunc(AlteraTamanhoJanela);

    glutMouseFunc(GerenciaMouse);
    glutPassiveMotionFunc(GerenciaMouseNav);
    if(HIDE_CURSOR){glutSetCursor(GLUT_CURSOR_NONE);}

    glutIgnoreKeyRepeat(1);

    glutKeyboardFunc(GerenciaTecladoDown);
    glutKeyboardUpFunc(GerenciaTecladoUp);

    glutSpecialFunc(GerenciaEspecialDown);
    glutSpecialUpFunc(GerenciaEspecialUp);

	Inicializa();
    
    glutMainLoop();
}

void Inicializa(){
    //Background
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f); //PRETO  (Fica mt bonito!!)
    //glClearColor(0.6f, 1.0f, 1.0f, 1.0f);   //Azulado

    //Ilumicação
	
        // Habilita o modelo de colorização de Gouraud
        glShadeModel(GL_SMOOTH);  // a cor de cada ponto da primitiva é interpolada a partir dos vértices
        //glShadeModel(GL_FLAT);  // a cor de cada primitiva é única em todos os pontos

        // Define a refletância do material 
        glMaterialfv(GL_FRONT, GL_SPECULAR, especularidade);
        // Define a concentração do brilho
        glMateriali(GL_FRONT, GL_SHININESS, especMaterial);

        // Ativa o uso da luz ambiente 
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, luzAmbiente);

        // Define os parâmetros da luz de número 0
        glLightfv(GL_LIGHT0, GL_AMBIENT, luzAmbiente); 
        glLightfv(GL_LIGHT0, GL_DIFFUSE, luzDifusa );
        glLightfv(GL_LIGHT0, GL_SPECULAR, luzEspecular );
        glLightfv(GL_LIGHT0, GL_POSITION, posicaoLuz );

        // Habilita a definição da cor do material a partir da cor corrente
        glEnable(GL_COLOR_MATERIAL);
        //Habilita o uso de iluminação
        glEnable(GL_LIGHTING);  
        // Habilita a luz de número 0
        glEnable(GL_LIGHT0);

    //Camera
	camAngle = 45;

    //Enable depth testing
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    //etc
    GerarMapa();
    Gamemode();
    DistribuirNuvens();

    Timer(1);
}

void Timer(int v){

    Hitbox();
    ApontarBloco();
    Animacao();
    Movimento();
	Camera();
	glutPostRedisplay();

    glutTimerFunc(1000/FPS, Timer, v);
}

void Desenha(){
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, largura, altura);

    DesenhaMapa();
    if(SHOW_CAMERA_HITBOX){DesenhaHitbox();}
    if(bloco_apontado[0] != -1){DesenhaBlocoApontado();}
    DesenhaNuvens();
    DesenhaItens();
    CalculaCurvas();
    
    glutSwapBuffers();
}

void AlteraTamanhoJanela(GLint largura, GLint altura){
	if(altura == 0) altura = 1;
	glViewport(0, 0, largura, altura);
	fAspect = (GLfloat)largura / (GLfloat)altura;
	Camera();
}

void Camera(){
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(camAngle, fAspect, 0.1, 4000);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    //Rotate normal
    float sensibility = 0.5;
    float n[16];

    //MAX mousey
    if(mousey * 360 * sensibility > 89.99){mousey = 89.99 / (360 * sensibility);}
    if(mousey * 360 * sensibility < -89.99){mousey = -89.99 / (360 * sensibility);}

    glPushMatrix();
        glTranslated(normal[0], normal[1], normal[2]);
        glRotatef(mousey * 360 * sensibility, 0, 1, 0);
        glRotatef(mousex * 360 * sensibility, 0, 0, 1);
        glGetFloatv(GL_MODELVIEW_MATRIX, n);
    glPopMatrix();

    normal[0] = n[0];
    normal[1] = n[4];
    normal[2] = n[8];

    //Normalize normal
    float norm = sqrt(pow(normal[0], 2) + pow(normal[1], 2) + pow(normal[2], 2));
    normal[0] /= norm;
    normal[1] /= norm;
    normal[2] /= norm;

    //Sense - walk
    float sense = 10;
    normal[0] *= sense;
    normal[1] *= sense;
    normal[2] *= sense;

    //Calculate normal90 = up x normal
    normal90[0] = up[1] * normal[2] - up[2] * normal[1];
    normal90[1] = up[2] * normal[0] - up[0] * normal[2];

    //Mova a camera e a normal para o ponto de visão (tamanho da hitbox da camera)
    float cam_view[3] = {0};
    float normal_view[3] = {0};
    cam_view[0] = cam[0];
    cam_view[1] = cam[1];
    cam_view[2] = cam[2] + CAMERA_HITBOX_SIZE * 2;
    normal_view[0] = normal[0] + cam[0];
    normal_view[1] = normal[1] + cam[1];
    normal_view[2] = normal[2] + cam[2] + CAMERA_HITBOX_SIZE * 2;
    
    //Look
    gluLookAt(
        cam_view[0], cam_view[1], cam_view[2],
        normal_view[0], normal_view[1], normal_view[2],
        up[0], up[1], up[2]
    );

    //Print coords
    //std::cout << "cam: " << cam[0]/BLOCK_SIZE << " " << cam[1]/BLOCK_SIZE << " " << cam[2]/BLOCK_SIZE << std::endl;
}

void Movimento(){
    //Walk Speed
    if(walk_speed > 0){
        walk_speed = walk_speed - 0.1;
    }else{
        walk_speed = 0;
    }

    if(keystates['w'] || keystates['a'] || keystates['s'] || keystates['d'] || specialstates[112] || keystates[32]){
        if(walk_speed < MAX_WALK_SPEED){
            walk_speed = walk_speed + WALK_ACCELERATION;
        }
    }

    //Camera Speed
    if(camera_speed > 0){
        camera_speed = camera_speed - 0.1;
    }else{
        camera_speed = 0;
    }

    if(specialstates[GLUT_KEY_UP] || specialstates[GLUT_KEY_DOWN] || specialstates[GLUT_KEY_LEFT] || specialstates[GLUT_KEY_RIGHT]){
        if(camera_speed < MAX_CAMERA_SPEED){
            camera_speed = camera_speed + CAMERA_ACCELERATION;
        }
    }

    //Walk - Normal resultante
    float normal_resultante[3] = {0,0,0};
    if(keystates['w']) {
        normal_resultante[0] += normal[0];
        normal_resultante[1] += normal[1];
        if(CAM_MODE == 1) {normal_resultante[2] += normal[2];};
    }
    if(keystates['s']) {
        normal_resultante[0] -= normal[0];
        normal_resultante[1] -= normal[1];
        if(CAM_MODE == 1) {normal_resultante[2] -= normal[2];}
    }
    if(keystates['a']) {
        normal_resultante[0] += normal90[0];
        normal_resultante[1] += normal90[1];
        if(CAM_MODE == 1) {normal_resultante[2] += normal90[2];}
    }
    if(keystates['d']) {
        normal_resultante[0] -= normal90[0];
        normal_resultante[1] -= normal90[1];
        if(CAM_MODE == 1) {normal_resultante[2] -= normal90[2];}
    }
    
    float norm_g = sqrt(pow(normal_resultante[0], 2) + pow(normal_resultante[1], 2) + pow(normal_resultante[2], 2));
    if(norm_g == 0){norm_g = 1;}
    normal_resultante[0] /= norm_g;
    normal_resultante[1] /= norm_g;
    normal_resultante[2] /= norm_g;

    normal_resultante[0] *= walk_speed * 10;
    normal_resultante[1] *= walk_speed * 10;
    normal_resultante[2] *= walk_speed * 10;

    //Shift - Down
    if(specialstates[112] && CAM_MODE == 1) {
        normal_resultante[2] -= MAX_WALK_SPEED * 5;
    }

    //Space - Up
    if(keystates[32] && CAM_MODE == 1) {
        normal_resultante[2] += MAX_WALK_SPEED * 5;
    }

    //Gravity
    if(GRAVITY){

        //Gravity Speed
        gravity_speed = gravity_speed + GRAVITY_ACCELERATION;
        
        if(gravity_speed > MAX_GRAVITY_SPEED){
            gravity_speed = MAX_GRAVITY_SPEED;
        }

        normal_resultante[2] -= gravity_speed;

        //Se a camera estiver no chão
        if(COLLISION && gravity_speed > 0){

            //Tem um bloco embaixo da camera?
            int block[3] = {0};
            block[0] = floor(cam[0] / BLOCK_SIZE);
            block[1] = floor(cam[1] / BLOCK_SIZE);
            block[2] = floor(cam[2] / BLOCK_SIZE) - 1;

            //O bloco pertence ao mapa?
            if(
                block[0] >= 0 && block[0] <= 63 &&
                block[1] >= 0 && block[1] <= 63 &&
                block[2] >= 0 && block[2] <= 63
            ){

                //O bloco é solido?
                if(mapa[block[0]][block[1]][block[2]] != 0 && mapa[block[0]][block[1]][block[2]] != 5){
                    normal_resultante[2] += gravity_speed;
                    gravity_speed = 0;
                }

            }

        }

    }

    //Space - Jump
    if(keystates[32] && gravity_speed == 0 && COLLISION){
        keystates[32] = false;
        gravity_speed = -JUMP_SPEED;
    }

    //Collision
    if(COLLISION){

        //Considerando a normal resultante, onde a camera vai estar no proximo frame?
        float cam_next[3] = {cam[0] + normal_resultante[0], cam[1] + normal_resultante[1], cam[2] + normal_resultante[2]};

        //Considerando a normal resultante, onde a hitbox da camera vai estar no proximo frame?
        float current_camera_hitbox_next[8][3] = {0};
        for(int i = 0; i < 8; i++){
            current_camera_hitbox_next[i][0] = current_camera_hitbox[i][0] + normal_resultante[0];
            current_camera_hitbox_next[i][1] = current_camera_hitbox[i][1] + normal_resultante[1];
            current_camera_hitbox_next[i][2] = current_camera_hitbox[i][2] + normal_resultante[2];
        }

        //Em qual bloco a camera vai estar no proximo frame?
        int block[3] = {0};
        block[0] = floor(cam_next[0] / BLOCK_SIZE);
        block[1] = floor(cam_next[1] / BLOCK_SIZE);
        block[2] = floor(cam_next[2] / BLOCK_SIZE);

        //Quais são os blocos em volta da camera no proximo frame?
        int blocks_around[7][3] = {
            {block[0], block[1], block[2]},
            {block[0] + 1, block[1], block[2]},
            {block[0] - 1, block[1], block[2]},
            {block[0], block[1] + 1, block[2]},
            {block[0], block[1] - 1, block[2]},
            {block[0], block[1], block[2] + 1},
            {block[0], block[1], block[2] - 1}
        };

        //Para cada bloco em volta da camera no proximo frame
        bool collision[8] = {false};//Pontos da hitbox da camera que colidem com algum bloco
        for (int j = 0; j < 7; j++){

            //A resultante é nula?
            if(normal_resultante[0] == 0 && normal_resultante[1] == 0 && normal_resultante[2] == 0){
                break;
            }

            //O não está dentro do mapa?
            if(
                blocks_around[j][0] < 0 || blocks_around[j][0] > 63 ||
                blocks_around[j][1] < 0 || blocks_around[j][1] > 63 ||
                blocks_around[j][2] < 0 || blocks_around[j][2] > 63
            ){continue;}
            
            //Se o bloco for solido
            if(mapa[blocks_around[j][0]][blocks_around[j][1]][blocks_around[j][2]] == 0){continue;}

            //Se o bloco é água
            if(mapa[blocks_around[j][0]][blocks_around[j][1]][blocks_around[j][2]] == 5){continue;}

            //Para cada ponto da hitbox da camera no proximo frame
            for(int i = 0; i < 8; i++){

                bool block_face_intersection[6] = {false};

                //Se o ponto estiver dentro do bloco
                if(
                    current_camera_hitbox_next[i][0] > blocks_around[j][0] * BLOCK_SIZE &&
                    current_camera_hitbox_next[i][0] < blocks_around[j][0] * BLOCK_SIZE + BLOCK_SIZE &&
                    current_camera_hitbox_next[i][1] > blocks_around[j][1] * BLOCK_SIZE &&
                    current_camera_hitbox_next[i][1] < blocks_around[j][1] * BLOCK_SIZE + BLOCK_SIZE &&
                    current_camera_hitbox_next[i][2] > blocks_around[j][2] * BLOCK_SIZE &&
                    current_camera_hitbox_next[i][2] < blocks_around[j][2] * BLOCK_SIZE + BLOCK_SIZE 
                ){
                    collision[i] = true;

                    //Centro das faces
                    float cube_x = blocks_around[j][0] * BLOCK_SIZE;
                    float cube_y = blocks_around[j][1] * BLOCK_SIZE;
                    float cube_z = blocks_around[j][2] * BLOCK_SIZE;

                    float faces[6][3] = {
                        {cube_x, cube_y + BLOCK_SIZE/2, cube_z + BLOCK_SIZE/2},
                        {cube_x + BLOCK_SIZE, cube_y + BLOCK_SIZE/2, cube_z + BLOCK_SIZE/2},
                        {cube_x + BLOCK_SIZE/2, cube_y, cube_z + BLOCK_SIZE/2},
                        {cube_x + BLOCK_SIZE/2, cube_y + BLOCK_SIZE, cube_z + BLOCK_SIZE/2},
                        {cube_x + BLOCK_SIZE/2, cube_y + BLOCK_SIZE/2, cube_z},
                        {cube_x + BLOCK_SIZE/2, cube_y + BLOCK_SIZE/2, cube_z + BLOCK_SIZE}
                    };

                    //Para cada face do bloco
                    float cam_face_distance[6] = {0};
                    float cam_face_distance_next[6] = {0};
                    for (int k = 0; k < 6; k++){

                        //Distancia da camera até o centro da face
                        cam_face_distance[k] = sqrt(
                            pow(cam[0] - faces[k][0], 2) +
                            pow(cam[1] - faces[k][1], 2) +
                            pow(cam[2] - faces[k][2], 2)
                        );

                        //Distancia da camera até o centro da face
                        cam_face_distance_next[k] = sqrt(
                            pow(cam_next[0] - faces[k][0], 2) +
                            pow(cam_next[1] - faces[k][1], 2) +
                            pow(cam_next[2] - faces[k][2], 2)
                        );

                    }

                    //A face mais próxima do bloco
                    int face_index = 0;
                    for (int k = 0; k < 6; k++){
                        if(cam_face_distance[k] < cam_face_distance[face_index]){
                            face_index = k;
                        }
                    }

                    int face_index_next = 0;
                    for (int k = 0; k < 6; k++){
                        if(cam_face_distance_next[k] < cam_face_distance_next[face_index_next]){
                            face_index_next = k;
                        }
                    }

                    //A face do bloco atravessada
                    block_face_intersection[face_index] = true;
                    //block_face_intersection[face_index_next] = true;

                    //Bloqueia o movimento na direção da face do bloco atravessada
                    if(block_face_intersection[0]){
                        normal_resultante[0] = 0;
                    }
                    if(block_face_intersection[1]){
                        normal_resultante[0] = 0;
                    }
                    if(block_face_intersection[2]){
                        normal_resultante[1] = 0;
                    }
                    if(block_face_intersection[3]){
                        normal_resultante[1] = 0;
                    }
                    if(block_face_intersection[4]){
                        normal_resultante[2] = 0;
                    }
                    if(block_face_intersection[5]){
                        normal_resultante[2] = 0;
                    }

                    //printf("--------------------\n");
                    //std::cout << "block: " << blocks_around[j][0] << " " << blocks_around[j][1] << " " << blocks_around[j][2] << std::endl;
                    //std::cout << "block_face_intersection: " << block_face_intersection[0] << block_face_intersection[1] << block_face_intersection[2] << block_face_intersection[3] << block_face_intersection[4] << block_face_intersection[5] << std::endl;
                    //std::cout << "normal_resultante: " << normal_resultante[0] << " " << normal_resultante[1] << " " << normal_resultante[2] << std::endl;

                }

                if(block_face_intersection[0] || block_face_intersection[1] || block_face_intersection[2] || block_face_intersection[3] || block_face_intersection[4] || block_face_intersection[5]){
                    break;
                }

            }

        }

    }

    //Walk
    cam[0] += normal_resultante[0];
    cam[1] += normal_resultante[1];
    cam[2] += normal_resultante[2];

    //Especial
    float sensibility_special = 0.01 * camera_speed;
    if(specialstates[GLUT_KEY_UP]) {
        mousey += sensibility_special;
    }
    if(specialstates[GLUT_KEY_DOWN]) {
        mousey -= sensibility_special;
    }
    if(specialstates[GLUT_KEY_LEFT]) {
        mousex -= sensibility_special;
    }
    if(specialstates[GLUT_KEY_RIGHT]) {
        mousex += sensibility_special;
    }
}

void Animacao(){
    //NUVENS

    //Adicione a normal aos pontos das nuvens
    float nuvem_normal[3] = {0,1,0};
    for(int i = 0; i < nuvens_limite; i++){
        if(nuvens[i][2] == 0){continue;}

        nuvens[i][0] += nuvem_normal[0] * nuvens_speed;
        nuvens[i][1] += nuvem_normal[1] * nuvens_speed;
        nuvens[i][2] += nuvem_normal[2] * nuvens_speed;
    }

    //Se alguma nuvem sair do mapa (MAP LENGTH), remova
    for(int i = 0; i < nuvens_limite; i++){
       
        if(
            nuvens[i][0] < 0 || nuvens[i][0] > MAP_LENGTH  ||
            nuvens[i][1] < 0 || nuvens[i][1] > MAP_LENGTH
        ){
            nuvens[i][0] = 0;
            nuvens[i][1] = 0;
            nuvens[i][2] = 0;
        }

    }

    //Gere novas nuvens
    for(int i = 0; i < MAP_LENGTH; i++){
        
        bool sera_gerada = rand() % 100 < nuvens_rate;
        if(!sera_gerada){continue;}

        //Encontre um slot
        for(int j = 0; j < nuvens_limite; j++){
            if(nuvens[j][2] == 0){
                nuvens[j][0] = i;
                nuvens[j][1] = 0;
                nuvens[j][2] = nuvem_altura + (rand() % delta_altura)/10;
                break;
            }

        }

    }

    //ITENS

    //Avance a animação dos itens
    for(int i = 0; i < itens_limite; i++){
        
        if(itens[i][3] != 0 ){
            itens[i][4]++;
        }

    }

    //Se a animação dos itens acabar, remova
    int MAX_ITEM_ANIMATION = 1000;
    for(int i = 0; i < itens_limite; i++){
        
        if(itens[i][4] > MAX_ITEM_ANIMATION){
            itens[i][0] = 0;
            itens[i][1] = 0;
            itens[i][2] = 0;
            itens[i][3] = 0;
            itens[i][4] = 0;
        }
        
    }

    //Gravidade
    if(GRAVITY){

        float item_speed = 0.1;
            
        //Para cada item
        for(int i = 0; i < itens_limite; i++){

            //O item é válido?
            if(itens[i][3] == 0){continue;}
            itens[i][2] -= item_speed;

            //Há um chão abaixo do item?
            int block[3] = {0};
            block[0] = floor(itens[i][0]);
            block[1] = floor(itens[i][1]);
            block[2] = floor(itens[i][2]) - 1;

            //O bloco pertence ao mapa?
            if(
                block[0] >= 0 && block[0] <= 63 &&
                block[1] >= 0 && block[1] <= 63 &&
                block[2] >= 0 && block[2] <= 63
            ){

                //O bloco é solido?
                if(
                    mapa[block[0]][block[1]][block[2]] != 0 && 
                    mapa[block[0]][block[1]][block[2]] != 5 &&
                    abs(block[2]-itens[i][2]) < 1.8
                ){
                    itens[i][2] += item_speed;
                }

            }
            
        }

    }
        
    //Destroi perto do player
    float destroy_range = 25;
    for(int i = 0; i < itens_limite; i++){
        if(itens[i][3] == 0){continue;}

        float distance = sqrt(
            pow(itens[i][0]*BLOCK_SIZE - cam[0], 2) +
            pow(itens[i][1]*BLOCK_SIZE - cam[1], 2) +
            pow(itens[i][2]*BLOCK_SIZE - cam[2], 2)
        );

        if(distance < destroy_range){
            itens[i][0] = 0;
            itens[i][1] = 0;
            itens[i][2] = 0;
            itens[i][3] = 0;
            itens[i][4] = 0;
        }

    }

}

void ApontarBloco(){
    //Range máxima
    int range = 10;

    //Dada a posição da camera e a normal, qual o primeiro bloco que a normal atravessa?
    float inicio[3] = {cam[0], cam[1], cam[2]};
    float direcao[3] = {normal[0], normal[1], normal[2]};

    //Normaliza a direção
    float norm = sqrt(pow(direcao[0], 2) + pow(direcao[1], 2) + pow(direcao[2], 2));
    direcao[0] /= norm;
    direcao[1] /= norm;
    direcao[2] /= norm;

    //Função de reta
    for(int i = 0; i < range; i++){
        int x = round((inicio[0] + direcao[0] * i * BLOCK_SIZE) / BLOCK_SIZE);
        int y = round((inicio[1] + direcao[1] * i * BLOCK_SIZE) / BLOCK_SIZE);
        int z = round((inicio[2] + direcao[2] * i * BLOCK_SIZE) / BLOCK_SIZE);

        //Se o bloco estiver dentro do mapa
        if(
            x >= 0 && x <= 63 &&
            y >= 0 && y <= 63 &&
            z >= 0 && z <= 63
        ){
            //Se o bloco for solido
            if(mapa[x][y][z] != 0){
                bloco_apontado[0] = x;
                bloco_apontado[1] = y;
                bloco_apontado[2] = z;
                break;
            }

            //Anterior
            bloco_apontado_anterior[0] = x;
            bloco_apontado_anterior[1] = y;
            bloco_apontado_anterior[2] = z;
        }

        bloco_apontado[0] = -1;
        bloco_apontado[1] = -1;
        bloco_apontado[2] = -1;

    }

}

void Hitbox(){
    //Calcula a hitbox da camera
    current_camera_hitbox[0][0] = CAMERA_HITBOX_SIZE + cam[0];
    current_camera_hitbox[0][1] = CAMERA_HITBOX_SIZE + cam[1];
    current_camera_hitbox[0][2] = CAMERA_HITBOX_SIZE + cam[2];

    current_camera_hitbox[1][0] = -CAMERA_HITBOX_SIZE + cam[0];
    current_camera_hitbox[1][1] = CAMERA_HITBOX_SIZE + cam[1];
    current_camera_hitbox[1][2] = CAMERA_HITBOX_SIZE + cam[2];

    current_camera_hitbox[2][0] = -CAMERA_HITBOX_SIZE + cam[0];
    current_camera_hitbox[2][1] = -CAMERA_HITBOX_SIZE + cam[1];
    current_camera_hitbox[2][2] = CAMERA_HITBOX_SIZE + cam[2];

    current_camera_hitbox[3][0] = CAMERA_HITBOX_SIZE + cam[0];
    current_camera_hitbox[3][1] = -CAMERA_HITBOX_SIZE + cam[1];
    current_camera_hitbox[3][2] = CAMERA_HITBOX_SIZE + cam[2];

    current_camera_hitbox[4][0] = CAMERA_HITBOX_SIZE + cam[0];
    current_camera_hitbox[4][1] = CAMERA_HITBOX_SIZE + cam[1];
    current_camera_hitbox[4][2] = -CAMERA_HITBOX_SIZE + cam[2];

    current_camera_hitbox[5][0] = -CAMERA_HITBOX_SIZE + cam[0];
    current_camera_hitbox[5][1] = CAMERA_HITBOX_SIZE + cam[1];
    current_camera_hitbox[5][2] = -CAMERA_HITBOX_SIZE + cam[2];

    current_camera_hitbox[6][0] = -CAMERA_HITBOX_SIZE + cam[0];
    current_camera_hitbox[6][1] = -CAMERA_HITBOX_SIZE + cam[1];
    current_camera_hitbox[6][2] = -CAMERA_HITBOX_SIZE + cam[2];

    current_camera_hitbox[7][0] = CAMERA_HITBOX_SIZE + cam[0];
    current_camera_hitbox[7][1] = -CAMERA_HITBOX_SIZE + cam[1];
    current_camera_hitbox[7][2] = -CAMERA_HITBOX_SIZE + cam[2];
}

void Gamemode(){
    //Spectator
    if(GAME_MODE == 1){
        printf("GAMEMODE: Spectator\n");
        MAX_WALK_SPEED = 2;
        CAM_MODE = 1;
        GRAVITY = false;
        COLLISION = false;
    }

    //Survival
    if(GAME_MODE == 0){
        printf("GAMEMODE: Survival\n");
        MAX_WALK_SPEED = 1;
        CAM_MODE = 0;
        GRAVITY = true;
        COLLISION = true;
    }

    if(debug == true){
        printf("DEBUG: ON\n");
        SHOW_BEZIER_CURVES = true;
        SHOW_CAMERA_HITBOX = true;
    }

    if(debug == false){
        printf("DEBUG: OFF\n");
        SHOW_BEZIER_CURVES = false;
        SHOW_CAMERA_HITBOX = false;
    }

}

void RecriarMapa(){
    printf("REGENERATING WORLD\n");

    //Nova seed com time
    MAP_SEED = time(NULL) % 1000000 + rand() % 1000000;
    printf("MAP_SEED: %d\n", MAP_SEED);

    LimparMapa();
    GerarMapa();
    DistribuirNuvens();
}

void GerenciaMouse(int button, int state, int x, int y){
    //printf("MOUSE: %d %d %d %d\n", button, state, x, y);

    //Clique esquerdo
    if(button == GLUT_LEFT_BUTTON && state == GLUT_DOWN){
        if(bloco_apontado[0] != -1){

            //Adiciona ao itens
            AdicionarItem(bloco_apontado[0], bloco_apontado[1], bloco_apontado[2], mapa[bloco_apontado[0]][bloco_apontado[1]][bloco_apontado[2]]);

            mapa[bloco_apontado[0]][bloco_apontado[1]][bloco_apontado[2]] = 0;
        }
    }

    //Clique direito
    if(button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN){
        if(bloco_apontado_anterior[0] != -1 && bloco_apontado[0] != -1){
            mapa[bloco_apontado_anterior[0]][bloco_apontado_anterior[1]][bloco_apontado_anterior[2]] = bloco_selecionado;
        }
    }

    //Define bloco selecionado com o scroll
    if(button == 3 || button == 4){
        if(button == 3){bloco_selecionado++;}
        if(button == 4){bloco_selecionado--;}

        if(bloco_selecionado > 9){bloco_selecionado = 1;}
        if(bloco_selecionado < 1){bloco_selecionado = 9;}

        switch (bloco_selecionado)
        {
        case 1:
            printf("BLOCK SELECTED: Grass\n");
            break;

        case 2:
            printf("BLOCK SELECTED: Dirt\n");
            break;
        
        case 3:
            printf("BLOCK SELECTED: Stone\n");
            break;

        case 4:
            printf("BLOCK SELECTED: Bedrock\n");
            break;

        case 5:
            printf("BLOCK SELECTED: Water\n");
            break;

        case 6:
            printf("BLOCK SELECTED: Wood\n");
            break;

        case 7:
            printf("BLOCK SELECTED: Leaves\n");
            break;

        case 8:
            printf("BLOCK SELECTED: Sand\n");
            break;

        case 9:
            printf("BLOCK SELECTED: Sphere\n");
            break;

        default:
            break;
        }

    }

}

void GerenciaMouseNav(int x, int y){
    //printf("MOUSE nav: %d %d\n", x, y);

    if(!ENABLE_MOUSE_NAV){return;}

    float sensibility = (((float)largura)/20000000) * MOUSE_SENSE;

    if(x > largura/2){
        mousex += (x - largura/2) * sensibility;
    }

    if(x < largura/2){
        mousex -= (largura/2 - x) * sensibility;
    }

    if(y > altura/2){
        mousey -= (y - altura/2) * sensibility;
    }

    if(y < altura/2){
        mousey += (altura/2 - y) * sensibility;
    }

    //Se sair do centro da tela
    if(x != largura/2 || y != altura/2){

        //Move o mouse de volta para o centro da tela
        glutWarpPointer(largura/2, altura/2);

    }

}

void GerenciaTecladoDown(unsigned char key, int x, int y) {
    if (key == 'w' || key == 'W') {
        keystates['w'] = true;
    }
    if (key == 's' || key == 'S') {
        keystates['s'] = true;
    }
    if (key == 'a' || key == 'A') {
        keystates['a'] = true;
    }
    if (key == 'd' || key == 'D') {
        keystates['d'] = true;
    }

    if(key == ' '){ //Space
        keystates[32] = true;
    }

    //Configs

    //G - Switch game mode
    if(key == 'g' || key == 'G'){
        if(GAME_MODE == 0){
            GAME_MODE = 1;
        }else{
            GAME_MODE = 0;
        }
        Gamemode();
    }

    //R - Regenerate world
    if(key == 'r' || key == 'R'){
        RecriarMapa();
    }

    //P - Debug
    if(key == 'p' || key == 'P'){
        if(debug == false){
            debug = true;
        }else{
            debug = false;
        }
        Gamemode();
    }
}

void GerenciaTecladoUp(unsigned char key, int x, int y) {
    if (key == 'w' || key == 'W') {
        keystates['w'] = false;
    }
    if (key == 's' || key == 'S') {
        keystates['s'] = false;
    }
    if (key == 'a' || key == 'A') {
        keystates['a'] = false;
    }
    if (key == 'd' || key == 'D') {
        keystates['d'] = false;
    }

    if(key == ' '){ //Space
        keystates[32] = false;
    }
}

void GerenciaEspecialDown(int key, int x, int y) {

    if(key == GLUT_KEY_UP) {
        specialstates[GLUT_KEY_UP] = true;
    }
    if(key == GLUT_KEY_DOWN) {
        specialstates[GLUT_KEY_DOWN] = true;
    }
    if(key == GLUT_KEY_LEFT) {
        specialstates[GLUT_KEY_LEFT] = true;
    }
    if(key == GLUT_KEY_RIGHT) {
        specialstates[GLUT_KEY_RIGHT] = true;
    }

    if(key == 112) { //Shift
        specialstates[112] = true;
    }
}

void GerenciaEspecialUp(int key, int x, int y) {
    if(key == GLUT_KEY_UP) {
        specialstates[GLUT_KEY_UP] = false;
    }
    if(key == GLUT_KEY_DOWN) {
        specialstates[GLUT_KEY_DOWN] = false;
    }
    if(key == GLUT_KEY_LEFT) {
        specialstates[GLUT_KEY_LEFT] = false;
    }
    if(key == GLUT_KEY_RIGHT) {
        specialstates[GLUT_KEY_RIGHT] = false;
    }

    if(key == 112) { //Shift
        specialstates[112] = false;
    }
}

//Specific functions
void DesenhaCubo(int x, int y, int z, int tipo){
    //BLOCK_SIZE é o tamanho da aresta do cubo
    //Suas dimensões vão de x*BLOCK_SIZE até x*BLOCK_SIZE + BLOCK_SIZE

    //Otimizações
    //Este cubo está cercado por outros cubos diferentes de 0?
    int blocos_ao_redor[6][3] = {
        {x + 1, y, z},
        {x - 1, y, z},
        {x, y + 1, z},
        {x, y - 1, z},
        {x, y, z + 1},
        {x, y, z - 1}
    };
    
    //O tipo define a cor do cubo
    float color[3] = {0};

    switch (tipo)
    {
    case 1: //Grass
        color[0] = 0.1f;
        color[1] = 0.9f;
        color[2] = 0.1f;
        break;

    case 2: //Dirt
        color[0] = 0.5f;
        color[1] = 0.3f;
        color[2] = 0.1f;
        break;

    case 3: //Stone
        color[0] = 0.5f;
        color[1] = 0.5f;
        color[2] = 0.5f;
        break;

    case 4: //Bedrock
        color[0] = 0.1f;
        color[1] = 0.1f;
        color[2] = 0.1f;
        break;

    case 5: //Water
        color[0] = 0.3f;
        color[1] = 0.3f;
        color[2] = 0.9f;
        break;

    case 6: //Wood
        color[0] = 0.3f;
        color[1] = 0.3f;
        color[2] = 0.1f;
        break;

    case 7: //Leaves
        color[0] = 0.0f;
        color[1] = 0.5f;
        color[2] = 0.0f;
        break;

    case 8: //Sand
        color[0] = 1.0f;
        color[1] = 1.0f;
        color[2] = 0.4f;
        break;

    case 9: //SPHERE
        color[0] = 1.0f;
        color[1] = 1.0f;
        color[2] = 1.0f;
        break;
    
    default:
        color[0] = 1.0f;
        color[1] = 1.0f;
        color[2] = 1.0f;
        break;
    }

    //Esfera
    if(tipo == 9){
        glPushMatrix();

            glColor3f(color[0], color[1], color[2]);
            glTranslatef(BLOCK_SIZE * x + BLOCK_SIZE/2, BLOCK_SIZE * y + BLOCK_SIZE/2, BLOCK_SIZE * z + BLOCK_SIZE/2);
            glutSolidSphere(BLOCK_SIZE/2, 50, 50);

        glPopMatrix();

        return;
    }

    //Cubo
    glPushMatrix();

        //Se a face do bloco estiver em contato com o ar/água, desenhe
        if( (blocos_ao_redor [5][0] < 0 || blocos_ao_redor [5][0] > 63) ||
            mapa[blocos_ao_redor [5][0]][blocos_ao_redor [5][1]][blocos_ao_redor [5][2]] == 0 ||
            mapa[blocos_ao_redor [5][0]][blocos_ao_redor [5][1]][blocos_ao_redor [5][2]] == 5
        ){
            glBegin(GL_QUADS); //Bottom

            glNormal3f(0.0f, 0.0f, 1.0f);//?
            glColor3f(color[0], color[1], color[2]);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y, BLOCK_SIZE * z);
            glVertex3f(BLOCK_SIZE * x + BLOCK_SIZE, BLOCK_SIZE * y, BLOCK_SIZE * z);
            glVertex3f(BLOCK_SIZE * x + BLOCK_SIZE, BLOCK_SIZE * y + BLOCK_SIZE, BLOCK_SIZE * z);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y + BLOCK_SIZE, BLOCK_SIZE * z);

            glEnd();
        }

        if( (blocos_ao_redor [4][0] < 0 || blocos_ao_redor [4][0] > 63) ||
            mapa[blocos_ao_redor [4][0]][blocos_ao_redor [4][1]][blocos_ao_redor [4][2]] == 0 ||
            mapa[blocos_ao_redor [4][0]][blocos_ao_redor [4][1]][blocos_ao_redor [4][2]] == 5 
        ){
            glBegin(GL_QUADS); //Top

            glNormal3f(0.0f, 0.0f, 1.0f);
            glColor3f(color[0], color[1], color[2]);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y, BLOCK_SIZE * z + BLOCK_SIZE);
            glVertex3f(BLOCK_SIZE * x + BLOCK_SIZE, BLOCK_SIZE * y, BLOCK_SIZE * z + BLOCK_SIZE);
            glVertex3f(BLOCK_SIZE * x + BLOCK_SIZE, BLOCK_SIZE * y + BLOCK_SIZE, BLOCK_SIZE * z + BLOCK_SIZE);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y + BLOCK_SIZE, BLOCK_SIZE * z + BLOCK_SIZE);

            glEnd();
        }

        if( (blocos_ao_redor [3][0] < 0 || blocos_ao_redor [3][0] > 63) ||
            mapa[blocos_ao_redor [3][0]][blocos_ao_redor [3][1]][blocos_ao_redor [3][2]] == 0 ||
            mapa[blocos_ao_redor [3][0]][blocos_ao_redor [3][1]][blocos_ao_redor [3][2]] == 5 
        ){
            glBegin(GL_QUADS);

            glNormal3f(0.0f, -1.0f, 0.0f);
            glColor3f(color[0], color[1], color[2]);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y, BLOCK_SIZE * z);
            glVertex3f(BLOCK_SIZE * x + BLOCK_SIZE, BLOCK_SIZE * y, BLOCK_SIZE * z);
            glVertex3f(BLOCK_SIZE * x + BLOCK_SIZE, BLOCK_SIZE * y, BLOCK_SIZE * z + BLOCK_SIZE);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y, BLOCK_SIZE * z + BLOCK_SIZE);

            glEnd();
        }

        if( (blocos_ao_redor [2][0] < 0 || blocos_ao_redor [2][0] > 63) ||
            mapa[blocos_ao_redor [2][0]][blocos_ao_redor [2][1]][blocos_ao_redor [1][2]] == 0 ||
            mapa[blocos_ao_redor [2][0]][blocos_ao_redor [2][1]][blocos_ao_redor [1][2]] == 5 
        ){
            glBegin(GL_QUADS);

            glNormal3b(0.0f, 1.0f, 0.0f);
            glColor3f(color[0], color[1], color[2]);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y + BLOCK_SIZE, BLOCK_SIZE * z);
            glVertex3f(BLOCK_SIZE * x + BLOCK_SIZE, BLOCK_SIZE * y + BLOCK_SIZE, BLOCK_SIZE * z);
            glVertex3f(BLOCK_SIZE * x + BLOCK_SIZE, BLOCK_SIZE * y + BLOCK_SIZE, BLOCK_SIZE * z + BLOCK_SIZE);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y + BLOCK_SIZE, BLOCK_SIZE * z + BLOCK_SIZE);

            glEnd();
        }

        if( (blocos_ao_redor [0][0] < 0 || blocos_ao_redor [0][0] > 63) ||
            mapa[blocos_ao_redor [0][0]][blocos_ao_redor [0][1]][blocos_ao_redor [0][2]] == 0 ||
            mapa[blocos_ao_redor [0][0]][blocos_ao_redor [0][1]][blocos_ao_redor [0][2]] == 5 
        ){
            glBegin(GL_QUADS);

            glNormal3b(1.0f, 0.0f, 0.0f);
            glColor3f(color[0], color[1], color[2]);
            glVertex3f(BLOCK_SIZE * x + BLOCK_SIZE, BLOCK_SIZE * y, BLOCK_SIZE * z);
            glVertex3f(BLOCK_SIZE * x + BLOCK_SIZE, BLOCK_SIZE * y, BLOCK_SIZE * z + BLOCK_SIZE);
            glVertex3f(BLOCK_SIZE * x + BLOCK_SIZE, BLOCK_SIZE * y + BLOCK_SIZE, BLOCK_SIZE * z + BLOCK_SIZE);
            glVertex3f(BLOCK_SIZE * x + BLOCK_SIZE, BLOCK_SIZE * y + BLOCK_SIZE, BLOCK_SIZE * z);

            glEnd();
        }

        if ( (blocos_ao_redor [1][0] < 0 || blocos_ao_redor [1][0] > 63) ||
            mapa[blocos_ao_redor [1][0]][blocos_ao_redor [1][1]][blocos_ao_redor [1][2]] == 0 ||
            mapa[blocos_ao_redor [1][0]][blocos_ao_redor [1][1]][blocos_ao_redor [1][2]] == 5
        ){
            glBegin(GL_QUADS);

            glNormal3f(-1.0f, 0.0f, 0.0f);
            glColor3f(color[0], color[1], color[2]);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y, BLOCK_SIZE * z);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y, BLOCK_SIZE * z + BLOCK_SIZE);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y + BLOCK_SIZE, BLOCK_SIZE * z + BLOCK_SIZE);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y + BLOCK_SIZE, BLOCK_SIZE * z);

            glEnd();
        }

    glPopMatrix();

}

void DesenhaBlocoApontado(){
    //Faz um wireframe do bloco apontado
    glPushMatrix();

        //Se o bloco apontado estiver dentro do mapa
        if(
            bloco_apontado[0] >= 0 && bloco_apontado[0] <= 63 &&
            bloco_apontado[1] >= 0 && bloco_apontado[1] <= 63 &&
            bloco_apontado[2] >= 0 && bloco_apontado[2] <= 63
        ){

            //Make bolder
            glLineWidth(8);
            glColor3f(0.0f, 0.0f, 0.0f);
            glTranslatef(bloco_apontado[0] * BLOCK_SIZE + BLOCK_SIZE/2, bloco_apontado[1] * BLOCK_SIZE + BLOCK_SIZE/2, bloco_apontado[2] * BLOCK_SIZE + BLOCK_SIZE/2);
            glutWireCube(BLOCK_SIZE+1);
        }

    glPopMatrix();

    if(SHOW_RIGHT_CLICK){

        //Se há bloco apontada, desenhar bloco apontado anterior
        glPushMatrix();

            //Se o bloco apontado anterior estiver dentro do mapa
            if(
                bloco_apontado_anterior[0] >= 0 && bloco_apontado_anterior[0] <= 63 &&
                bloco_apontado_anterior[1] >= 0 && bloco_apontado_anterior[1] <= 63 &&
                bloco_apontado_anterior[2] >= 0 && bloco_apontado_anterior[2] <= 63
            ){

                //Make bolder
                glLineWidth(2);
                glColor3f(0.0f, 0.0f, 0.0f);
                glTranslatef(bloco_apontado_anterior[0] * BLOCK_SIZE + BLOCK_SIZE/2, bloco_apontado_anterior[1] * BLOCK_SIZE + BLOCK_SIZE/2, bloco_apontado_anterior[2] * BLOCK_SIZE + BLOCK_SIZE/2);
                glutWireCube(BLOCK_SIZE+1);
            }

        glPopMatrix();

    }

}

void DesenhaNuvens(){
    float tamanho = 3*BLOCK_SIZE;

    for(int i = 0; i < 100; i++){

        if(nuvens[i][2] == 0){continue;}

        glPushMatrix();

            glColor3f(1.0f, 1.0f, 1.0f);
            glTranslatef(nuvens[i][0]*BLOCK_SIZE + BLOCK_SIZE/2, nuvens[i][1]*BLOCK_SIZE + BLOCK_SIZE/2, nuvens[i][2]*BLOCK_SIZE + BLOCK_SIZE/2);
            
            //Um prisma
            glBegin(GL_QUADS); //TOP
                glNormal3f(0.0f, 0.0f, 1.0f);
                glVertex3f(-tamanho, -tamanho, tamanho/2);
                glVertex3f(tamanho, -tamanho, tamanho/2);
                glVertex3f(tamanho, tamanho, tamanho/2);
                glVertex3f(-tamanho, tamanho, tamanho/2);
            glEnd();

            glBegin(GL_QUADS); //BOTTOM
                glNormal3f(0.0f, 0.0f, -1.0f);
                glVertex3f(-tamanho, -tamanho, 0);
                glVertex3f(tamanho, -tamanho, 0);
                glVertex3f(tamanho, tamanho, 0);
                glVertex3f(-tamanho, tamanho, 0);
            glEnd();

            glBegin(GL_QUADS); //LEFT
                glNormal3f(-1.0f, 0.0f, 0.0f);
                glVertex3f(-tamanho, -tamanho, 0);
                glVertex3f(-tamanho, tamanho, 0);
                glVertex3f(-tamanho, tamanho, tamanho/2);
                glVertex3f(-tamanho, -tamanho, tamanho/2);
            glEnd();

            glBegin(GL_QUADS); //RIGHT
                glNormal3f(1.0f, 0.0f, 0.0f);
                glVertex3f(tamanho, -tamanho, 0);
                glVertex3f(tamanho, tamanho, 0);
                glVertex3f(tamanho, tamanho, tamanho/2);
                glVertex3f(tamanho, -tamanho, tamanho/2);
            glEnd();

            glBegin(GL_QUADS); //FRONT
                glNormal3f(0.0f, 1.0f, 0.0f);
                glVertex3f(-tamanho, tamanho, 0);
                glVertex3f(tamanho, tamanho, 0);
                glVertex3f(tamanho, tamanho, tamanho/2);
                glVertex3f(-tamanho, tamanho, tamanho/2);
            glEnd();

            glBegin(GL_QUADS); //BACK
                glNormal3f(0.0f, -1.0f, 0.0f);
                glVertex3f(-tamanho, -tamanho, 0);
                glVertex3f(tamanho, -tamanho, 0);
                glVertex3f(tamanho, -tamanho, tamanho/2);
                glVertex3f(-tamanho, -tamanho, tamanho/2);
            glEnd();

        glPopMatrix();
    }
    
}

void DesenhaItens(){
    for(int i = 0; i < 100; i++){

        if(itens[i][3] == 0){continue;}

        float x = itens[i][0];
        float y = itens[i][1];
        float z = itens[i][2];

        //O tipo define a cor do cubo
        float color[3] = {0};
        switch ((int) itens[i][3]){
            
            case 1: //Grass
                color[0] = 0.1f;
                color[1] = 0.9f;
                color[2] = 0.1f;
                break;

            case 2: //Dirt
                color[0] = 0.5f;
                color[1] = 0.3f;
                color[2] = 0.1f;
                break;

            case 3: //Stone
                color[0] = 0.5f;
                color[1] = 0.5f;
                color[2] = 0.5f;
                break;

            case 4: //Bedrock
                color[0] = 0.1f;
                color[1] = 0.1f;
                color[2] = 0.1f;
                break;

            case 5: //Water
                color[0] = 0.3f;
                color[1] = 0.3f;
                color[2] = 0.9f;
                break;

            case 6: //Wood
                color[0] = 0.3f;
                color[1] = 0.3f;
                color[2] = 0.1f;
                break;

            case 7: //Leaves
                color[0] = 0.0f;
                color[1] = 0.5f;
                color[2] = 0.0f;
                break;

            case 8: //Sand
                color[0] = 1.0f;
                color[1] = 1.0f;
                color[2] = 0.4f;
                break;
            
            default:
                color[0] = 1.0f;
                color[1] = 1.0f;
                color[2] = 1.0f;
                break;
        }
            
        //Cubo
        glPushMatrix();

            //Move the center of the cube to the center of the block
            glTranslatef(BLOCK_SIZE/2 - ITEM_SIZE/2, BLOCK_SIZE/2 - ITEM_SIZE/2, BLOCK_SIZE/2 - ITEM_SIZE/2);

            //Animate

                glTranslatef(x * BLOCK_SIZE + ITEM_SIZE/2, y * BLOCK_SIZE + ITEM_SIZE/2, z * BLOCK_SIZE + ITEM_SIZE/2);
                glRotatef(itens[i][4], 0.0f, 0.0f, 1.0f); //Rotate
                glTranslatef(0.0f, 0.0f, sin(itens[i][4]/10)); //Move up and down
                glTranslatef(-x * BLOCK_SIZE - ITEM_SIZE/2, -y * BLOCK_SIZE - ITEM_SIZE/2, -z * BLOCK_SIZE - ITEM_SIZE/2);


            glBegin(GL_QUADS); //Bottom

            glNormal3f(0.0f, 0.0f, 1.0f);//?
            glColor3f(color[0], color[1], color[2]);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y, BLOCK_SIZE * z);
            glVertex3f(BLOCK_SIZE * x + ITEM_SIZE, BLOCK_SIZE * y, BLOCK_SIZE * z);
            glVertex3f(BLOCK_SIZE * x + ITEM_SIZE, BLOCK_SIZE * y + ITEM_SIZE, BLOCK_SIZE * z);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y + ITEM_SIZE, BLOCK_SIZE * z);

            glEnd();
        
            glBegin(GL_QUADS); //Top

            glNormal3f(0.0f, 0.0f, 1.0f);
            glColor3f(color[0], color[1], color[2]);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y, BLOCK_SIZE * z + ITEM_SIZE);
            glVertex3f(BLOCK_SIZE * x + ITEM_SIZE, BLOCK_SIZE * y, BLOCK_SIZE * z + ITEM_SIZE);
            glVertex3f(BLOCK_SIZE * x + ITEM_SIZE, BLOCK_SIZE * y + ITEM_SIZE, BLOCK_SIZE * z + ITEM_SIZE);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y + ITEM_SIZE, BLOCK_SIZE * z + ITEM_SIZE);

            glEnd();

            glBegin(GL_QUADS);

            glNormal3f(0.0f, -1.0f, 0.0f);
            glColor3f(color[0], color[1], color[2]);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y, BLOCK_SIZE * z);
            glVertex3f(BLOCK_SIZE * x + ITEM_SIZE, BLOCK_SIZE * y, BLOCK_SIZE * z);
            glVertex3f(BLOCK_SIZE * x + ITEM_SIZE, BLOCK_SIZE * y, BLOCK_SIZE * z + ITEM_SIZE);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y, BLOCK_SIZE * z + ITEM_SIZE);

            glEnd();
        
            glBegin(GL_QUADS);

            glNormal3b(0.0f, 1.0f, 0.0f);
            glColor3f(color[0], color[1], color[2]);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y + ITEM_SIZE, BLOCK_SIZE * z);
            glVertex3f(BLOCK_SIZE * x + ITEM_SIZE, BLOCK_SIZE * y + ITEM_SIZE, BLOCK_SIZE * z);
            glVertex3f(BLOCK_SIZE * x + ITEM_SIZE, BLOCK_SIZE * y + ITEM_SIZE, BLOCK_SIZE * z + ITEM_SIZE);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y + ITEM_SIZE, BLOCK_SIZE * z + ITEM_SIZE);

            glEnd();
        
            glBegin(GL_QUADS);

            glNormal3b(1.0f, 0.0f, 0.0f);
            glColor3f(color[0], color[1], color[2]);
            glVertex3f(BLOCK_SIZE * x + ITEM_SIZE, BLOCK_SIZE * y, BLOCK_SIZE * z);
            glVertex3f(BLOCK_SIZE * x + ITEM_SIZE, BLOCK_SIZE * y, BLOCK_SIZE * z + ITEM_SIZE);
            glVertex3f(BLOCK_SIZE * x + ITEM_SIZE, BLOCK_SIZE * y + ITEM_SIZE, BLOCK_SIZE * z + ITEM_SIZE);
            glVertex3f(BLOCK_SIZE * x + ITEM_SIZE, BLOCK_SIZE * y + ITEM_SIZE, BLOCK_SIZE * z);

            glEnd();
        
            glBegin(GL_QUADS);

            glNormal3f(-1.0f, 0.0f, 0.0f);
            glColor3f(color[0], color[1], color[2]);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y, BLOCK_SIZE * z);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y, BLOCK_SIZE * z + ITEM_SIZE);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y + ITEM_SIZE, BLOCK_SIZE * z + ITEM_SIZE);
            glVertex3f(BLOCK_SIZE * x, BLOCK_SIZE * y + ITEM_SIZE, BLOCK_SIZE * z);

            glEnd();

        glPopMatrix();

    }

}

void DesenhaHitbox(){
    //Cubo usando current_camera_hitbox (wire)
    glPushMatrix();

        glColor3f(0.0f, 0.0f, 0.0f);
        glBegin(GL_LINES);

            glVertex3f(current_camera_hitbox[0][0], current_camera_hitbox[0][1], current_camera_hitbox[0][2]);
            glVertex3f(current_camera_hitbox[1][0], current_camera_hitbox[1][1], current_camera_hitbox[1][2]);

            glVertex3f(current_camera_hitbox[1][0], current_camera_hitbox[1][1], current_camera_hitbox[1][2]);
            glVertex3f(current_camera_hitbox[2][0], current_camera_hitbox[2][1], current_camera_hitbox[2][2]);

            glVertex3f(current_camera_hitbox[2][0], current_camera_hitbox[2][1], current_camera_hitbox[2][2]);
            glVertex3f(current_camera_hitbox[3][0], current_camera_hitbox[3][1], current_camera_hitbox[3][2]);

            glVertex3f(current_camera_hitbox[3][0], current_camera_hitbox[3][1], current_camera_hitbox[3][2]);
            glVertex3f(current_camera_hitbox[0][0], current_camera_hitbox[0][1], current_camera_hitbox[0][2]);

            glVertex3f(current_camera_hitbox[4][0], current_camera_hitbox[4][1], current_camera_hitbox[4][2]);
            glVertex3f(current_camera_hitbox[5][0], current_camera_hitbox[5][1], current_camera_hitbox[5][2]);

            glVertex3f(current_camera_hitbox[5][0], current_camera_hitbox[5][1], current_camera_hitbox[5][2]);
            glVertex3f(current_camera_hitbox[6][0], current_camera_hitbox[6][1], current_camera_hitbox[6][2]);

            glVertex3f(current_camera_hitbox[6][0], current_camera_hitbox[6][1], current_camera_hitbox[6][2]);
            glVertex3f(current_camera_hitbox[7][0], current_camera_hitbox[7][1], current_camera_hitbox[7][2]);

            glVertex3f(current_camera_hitbox[7][0], current_camera_hitbox[7][1], current_camera_hitbox[7][2]);
            glVertex3f(current_camera_hitbox[4][0], current_camera_hitbox[4][1], current_camera_hitbox[4][2]);

            glVertex3f(current_camera_hitbox[0][0], current_camera_hitbox[0][1], current_camera_hitbox[0][2]);
            glVertex3f(current_camera_hitbox[4][0], current_camera_hitbox[4][1], current_camera_hitbox[4][2]);

            glVertex3f(current_camera_hitbox[1][0], current_camera_hitbox[1][1], current_camera_hitbox[1][2]);
            glVertex3f(current_camera_hitbox[5][0], current_camera_hitbox[5][1], current_camera_hitbox[5][2]);

            glVertex3f(current_camera_hitbox[2][0], current_camera_hitbox[2][1], current_camera_hitbox[2][2]);
            glVertex3f(current_camera_hitbox[6][0], current_camera_hitbox[6][1], current_camera_hitbox[6][2]);

            glVertex3f(current_camera_hitbox[3][0], current_camera_hitbox[3][1], current_camera_hitbox[3][2]);
            glVertex3f(current_camera_hitbox[7][0], current_camera_hitbox[7][1], current_camera_hitbox[7][2]);

        glEnd();

    glPopMatrix();

}

void DesenhaMapa(){
    for(int x = 0; x < 64; x++){
        for(int y = 0; y < 64; y++){
            for(int z = 0; z < 64; z++){
                if(mapa[x][y][z] != 0){
                    DesenhaCubo(x, y, z, mapa[x][y][z]);
                }
            }
        }
    }

}

void GerarMapa(){
    //Parametros
    int altura_base = 32;
    int altura_pedra = 29;
    int altura_caverna = 32;
    int altura_agua = altura_base - 2;
    int min_limit = -3;
    int max_limit = 4;
    int spacing = 2;

    float smooth_out = 0.55;
    float smooth_out_chance = 1;
    float keep_terrain_level_chance = 0.75;

    float tree_chance = 0.03;

    //
    srand(MAP_SEED);

    int ints_aleat[64][64] = {0};
    for (int x = 0; x < MAP_LENGTH; x++){
        for (int y = 0; y < MAP_LENGTH; y++){

            //Quanto mais próximo da altura base, mais provavel
            ints_aleat[x][y] = altura_base + floor((rand() % (max_limit - min_limit + 1)) + min_limit);

        }
    }

    //Smooth out e keep terrain level
    for (int x = 0; x < MAP_LENGTH; x++){
        for (int y = 0; y < MAP_LENGTH; y++){

            //Chance de não suavizar
            if((rand() % 100) / 100.0 > smooth_out_chance){continue;}

            int blocos_ao_redor[4][2] = {
                {x + 1, y},
                {x - 1, y},
                {x, y + 1},
                {x, y - 1}
            };

            //Keep terrain level
            if((rand() % 100) / 100.0 < keep_terrain_level_chance){

                //Blocos adjacentes tornam-se iguais em altura ao bloco atual
                for ( int i = 0; i < 4; i++){
                    
                    //Bloco invalido?
                    if(
                        blocos_ao_redor[i][0] < 0 || blocos_ao_redor[i][0] > MAP_LENGTH ||
                        blocos_ao_redor[i][1] < 0 || blocos_ao_redor[i][1] > MAP_LENGTH
                    ){continue;}

                    ints_aleat[blocos_ao_redor[i][0]][blocos_ao_redor[i][1]] = ints_aleat[x][y];

                }

                continue;
            }
            
            //Para cada bloco ao redor válido
            float media_blocos = altura_base;
            int nro_blocos;
            for ( int i = 0; i < 4; i++){
                
                //Bloco invalido?
                if(
                    blocos_ao_redor[i][0] < 0 || blocos_ao_redor[i][0] > MAP_LENGTH ||
                    blocos_ao_redor[i][1] < 0 || blocos_ao_redor[i][1] > MAP_LENGTH
                ){continue;}

                nro_blocos++;
                media_blocos += ints_aleat[blocos_ao_redor[i][0]][blocos_ao_redor[i][1]];

            }
            media_blocos = media_blocos / nro_blocos;

            //floor(Altura do bloco * (1-smooth_out) + Media de alturas dos blocos ao redor) * smooth_out)
            ints_aleat[x][y] = floor(ints_aleat[x][y] * (1 - smooth_out) + media_blocos * smooth_out);

            //media com a altura base
            ints_aleat[x][y] = floor(ints_aleat[x][y] * (1 - smooth_out) + altura_base * smooth_out);

        }
        
    }

    //Altere os números de spacing em spacing para criar uma linha em Y
    for (int x = 0; x < MAP_LENGTH; x += spacing){

        int last_y = 0;
        for (int y = 0; y < MAP_LENGTH; y += spacing){
            
            //Formula de linha entre (x, y) e (x, last_y)
            int m = 1;
            if(y != last_y){m = floor((ints_aleat[x][y] - ints_aleat[x][last_y]) / (y - last_y));}

            int b = floor(ints_aleat[x][y] - m * y);

            for(int i = last_y; i < y; i++){
                ints_aleat[x][i] = floor(m * i + b);
                CriarPilastra(x, i, ints_aleat[x][i]);
            }

            last_y = y;
        }
    }

    //Altere os números de spacing em spacing para criar uma linha em X
    for (int x = 0; x < MAP_LENGTH; x += spacing){
        for (int y = 0; y < MAP_LENGTH; y++){

            //If z = 0, z = altura_base
            if(ints_aleat[x][y] == 0){ints_aleat[x][y] = altura_base;}
            if(ints_aleat[x + spacing][y] == 0){ints_aleat[x + spacing][y] = altura_base;}

            //Conecta os pontos (x, y) e (x + spacing, y)
            int m = 1;
            if(x != MAP_LENGTH - 1){m = floor((ints_aleat[x + spacing][y] - ints_aleat[x][y]) / (spacing));}

            int b = floor(ints_aleat[x][y] - m * x);

            for(int i = x; i < x + spacing; i++){
                ints_aleat[i][y] = floor(m * i + b);
                CriarPilastra(i, y, ints_aleat[i][y]);
            }
        }
    }

    //Substitui os blocos de terra e grama por pedra abaixo da altura_pedra
    for (int x = 0; x < MAP_LENGTH; x++){
        for (int y = 0; y < MAP_LENGTH; y++){
            for (int z = 0; z < altura_pedra; z++){
                if(mapa[x][y][z] == 2 || mapa[x][y][z] == 1){
                    mapa[x][y][z] = 3;
                }
            }
        }
    }

    //Cria o oceano
    CriarOceano(altura_agua, 0, MAP_LENGTH, 0, MAP_LENGTH);

    //Cria arvores
    for (int x = 0; x < MAP_LENGTH; x++){
        for (int y = 0; y < MAP_LENGTH; y++){

            //Chance de criar arvore
            if((rand() % 100) / 100.0 > tree_chance){continue;}

            //Se não é grama, pare
            if(mapa[x][y][ints_aleat[x][y]] != 1){continue;}

            //Se não tem espaço para a arvore, pare
            if(ints_aleat[x][y] + 7 >= 64){continue;}
            if(x-3 < 0 || x+3 >= MAP_LENGTH){continue;}
            if(y-3 < 0 || y+3 >= MAP_LENGTH){continue;}

            //Criar arvore
            CriarArvore(x - 3, y - 3, ints_aleat[x][y] + 1);
        }
    }
}

void LimparMapa(){
    for(int x = 0; x < 64; x++){
        for(int y = 0; y < 64; y++){
            for(int z = 0; z < 64; z++){
                mapa[x][y][z] = 0;
            }
        }
    }
}

void AdicionarItem(int x, int y, int z, int item){
    //From -BLOCK_SIZE/2 to BLOCK_SIZE/2 rand
    float item_delta = 20;

    //Procura um slot
    for(int i = 0; i < itens_limite; i++){

        //Se o slot estiver vazio
        if( itens[i][3] == 0){

            itens[i][0] = x + (rand() % (int) round(item_delta))/BLOCK_SIZE - (rand() % (int) round(item_delta))/BLOCK_SIZE;
            itens[i][1] = y + rand() % (int) round(item_delta)/BLOCK_SIZE - (rand() % (int) round(item_delta))/BLOCK_SIZE;
            itens[i][2] = z;
            itens[i][3] = item;
            itens[i][4] = 0; 
            return;

        }

    }

    return;
}

void DistribuirNuvens(){
    //Limpa
    LimparNuvens();

    //Distribui nuvens
    for(int i = 0; i < nuvens_limite; i++){
        nuvens[i][0] = rand() % (int) round(MAP_LENGTH);
        nuvens[i][1] = rand() % (int) round(MAP_LENGTH);

        nuvens[i][2] = nuvem_altura + (rand() % delta_altura)/10;
    }
}

void LimparNuvens(){
    for(int i = 0; i < nuvens_limite; i++){
        nuvens[i][0] = 0;
        nuvens[i][1] = 0;
        nuvens[i][2] = 0;
    }
}

void CriarPilastra(int x, int y, int z){
    //Bloco válido?
    if(
        x < 0 || x > 63 ||
        y < 0 || y > 63 ||
        z < 0 || z > 63
    ){return;}

    //Bloco inicial é grama
    mapa[x][y][z] = 1;

    //3 blocos de terra (ou até o chão)
    for(int i = 1; i <= 3; i++){
        if(z - i < 0){break;}
        mapa[x][y][z - i] = 2;
    }

    //X Bloco de pedra (dos 3 blocos de terra até z = 1)
    for(int i = 4; i <= z-1; i++){
        if(z - i < 0){break;}
        mapa[x][y][z - i] = 3;
    }

    //Bloco de bedrock (z = 0)
    mapa[x][y][0] = 4;
}

void CriarOceano(int startz, int startx, int endx, int starty, int endy){
    //Preencher todos os blocos não vazios <= z com agua
    for(int x = startx; x < endx; x++){
        for(int y = starty; y < endy; y++){
            for(int z = startz; z >= 0; z--){
                if(mapa[x][y][z] == 0){
                    mapa[x][y][z] = 5;

                    //Grama e Terra em contato com a água (se existirem e forem preenchidos) viram areia
                    int blocos_proximos[6][3] = {
                        {x + 1, y, z},
                        {x - 1, y, z},
                        {x, y + 1, z},
                        {x, y - 1, z},
                        {x, y, z + 1},
                        {x, y, z - 1}
                    };

                    for (int i = 0; i < 6; i++){
                        //Se o bloco for válido e não for vazio
                        if(
                            blocos_proximos[i][0] >= 0 && blocos_proximos[i][0] <= 63 &&
                            blocos_proximos[i][1] >= 0 && blocos_proximos[i][1] <= 63 &&
                            blocos_proximos[i][2] >= 0 && blocos_proximos[i][2] <= 63 &&
                            mapa[blocos_proximos[i][0]][blocos_proximos[i][1]][blocos_proximos[i][2]] != 0 &&
                            mapa[blocos_proximos[i][0]][blocos_proximos[i][1]][blocos_proximos[i][2]] != 5 &&
                            (
                                mapa[blocos_proximos[i][0]][blocos_proximos[i][1]][blocos_proximos[i][2]] == 1 || 
                                mapa[blocos_proximos[i][0]][blocos_proximos[i][1]][blocos_proximos[i][2]] == 2
                            )
                        ){
                            mapa[blocos_proximos[i][0]][blocos_proximos[i][1]][blocos_proximos[i][2]] = 8;
                        }
                    }
                }
            }
        }
    }
}

void CriarArvore(int x, int y, int z){
    //Arvore dataset (minecraft oak tree)
    int arvore[7][7][7] = {
        {
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 6, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0}
        },
        {
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 6, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0}
        },
        {
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 7, 7, 7, 0, 0},
            {0, 7, 7, 7, 7, 7, 0},
            {0, 7, 7, 6, 7, 7, 0},
            {0, 7, 7, 7, 7, 7, 0},
            {0, 0, 7, 7, 7, 0, 0},
            {0, 0, 0, 0, 0, 0, 0}
        },
        {
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 7, 7, 7, 0, 0},
            {0, 7, 7, 7, 7, 7, 0},
            {0, 7, 7, 6, 7, 7, 0},
            {0, 7, 7, 7, 7, 7, 0},
            {0, 0, 7, 7, 7, 0, 0},
            {0, 0, 0, 0, 0, 0, 0}
        },
        {
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 7, 7, 7, 0, 0},
            {0, 0, 7, 6, 7, 0, 0},
            {0, 0, 7, 7, 7, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0}
        },
        {
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 7, 0, 0, 0},
            {0, 0, 7, 7, 7, 0, 0},
            {0, 0, 0, 7, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0}
        },
        {
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0}
        }
    };

    //Cria a arvore
    for(int i = 0; i < 7; i++){
        for(int j = 0; j < 7; j++){
            for(int k = 0; k < 7; k++){
                if(arvore[i][j][k] != 0){
                    mapa[x + j][y + k][z + i] = arvore[i][j][k];
                }
            }
        }
    }
}

float CalculaPonto(char c, double t) {
    //Curva de bezier cúbica
	switch (c) {
		case 'x':
			return (pow(1 - t, 3) * Bx[0] + 3 * t * pow(1 - t, 2) * Bx[1] + 3 * pow(t, 2) * (1 - t) * Bx[2] + pow(t, 3) * Bx[3]);
			break;
		case 'y':
			return (pow(1 - t, 3) * By[0] + 3 * t * pow(1 - t, 2) * By[1] + 3 * pow(t, 2) * (1 - t) * By[2] + pow(t, 3) * By[3]);
			break;
		case 'z':
			return (pow(1 - t, 3) * Bz[0] + 3 * t * pow(1 - t, 2) * Bz[1] + 3 * pow(t, 2) * (1 - t) * Bz[2] + pow(t, 3) * Bz[3]);
			break;
        default:
            return 0;
            break;
	}
}

void CalculaCurvas(){ //Calcula e Mostra as curvas de bezier
    float Px, Py, Pz;

    for(int i = 0; i < itens_limite; i++){
        
        if(itens[i][3] == 0){continue;}
        itens_quantidade = 0;

        //Preencha todas as coordenadas com o item i
        for(int j = 0; j < 4; j++){
            Bx[j] = itens[i][0];
            By[j] = itens[i][1];
            Bz[j] = itens[i][2];
        }

        //Os 5 itens mais próximos do item i (REDUCED SEARCH)
        float menor_dist = 9999999;
        for(int j = 1; j < itens_limite; j++){
            
            if(itens[j][3] == 0){continue;}
            if(i == j){continue;}
            if(itens_quantidade == 4){break;}
            
            //distancia entre i e j
            float dist = sqrt(
                pow(itens[i][0] - itens[j][0], 2) +
                pow(itens[i][1] - itens[j][1], 2) +
                pow(itens[i][2] - itens[j][2], 2)
            );

            if(dist < menor_dist){
                menor_dist = dist;

                Bx[itens_quantidade] = itens[j][0];
                By[itens_quantidade] = itens[j][1];
                Bz[itens_quantidade] = itens[j][2];

                itens_quantidade++;
            }

        }

        //Coord 0 é a camera
        Bx[0] = cam[0]/BLOCK_SIZE;
        By[0] = cam[1]/BLOCK_SIZE;
        Bz[0] = cam[2]/BLOCK_SIZE;
        
        // Desenha a curva de bezier
        glColor3f(0.0f, 0.0f, 1.0f);
        glBegin(GL_LINE_STRIP);
            for (float t = 0; t <= 1; t=t+0.02) {
                Px = CalculaPonto('x', t);
                Py = CalculaPonto('y', t);
                Pz = CalculaPonto('z', t);

                if(SHOW_BEZIER_CURVES){
                    glVertex3d(Px * BLOCK_SIZE, Py * BLOCK_SIZE, Pz * BLOCK_SIZE);
                }

            }
        glEnd();

        // Desenha os pontos de controle
        if(SHOW_BEZIER_CURVES){
            glColor3f(1.0f, 0.06, 0.6f);
            glBegin(GL_LINE_STRIP);
                glVertex3i(Bx[0] * BLOCK_SIZE, By[0] * BLOCK_SIZE, Bz[0] * BLOCK_SIZE);
                glVertex3i(Bx[1] * BLOCK_SIZE, By[1] * BLOCK_SIZE, Bz[1] * BLOCK_SIZE);
                glVertex3i(Bx[2] * BLOCK_SIZE, By[2] * BLOCK_SIZE, Bz[2] * BLOCK_SIZE);
                glVertex3i(Bx[3] * BLOCK_SIZE, By[3] * BLOCK_SIZE, Bz[3] * BLOCK_SIZE);
            glEnd();
        }

        // MOVE OS ITENS

        //Distancia entre item e o jogador
        float dist = sqrt(
            pow(itens[i][0] - cam[0]/BLOCK_SIZE, 2) +
            pow(itens[i][1] - cam[1]/BLOCK_SIZE, 2) +
            pow(itens[i][2] - cam[2]/BLOCK_SIZE, 2)
        );

        //Dist em % da dist maxima
        float MAX_DIST = 3;
        float t = dist / MAX_DIST;

        if(t < 1){

            Px = CalculaPonto('x', t);
            Py = CalculaPonto('y', t);
            Pz = CalculaPonto('z', t);

            itens[i][0] = Px;
            itens[i][1] = Py;
            itens[i][2] = Pz;

        }
    }
}