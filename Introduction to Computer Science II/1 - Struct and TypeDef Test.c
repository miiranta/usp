#include <stdio.h>

//Makes the structure "a", "b" and "c", "a" and "b" are inside "c"
struct a{
    int HelloHello;
    int YouSayGoodbye;
    int ISayHello;
};

struct b{
    int HelloHello;
    int YouSayGoodbye;
    int ISayHello;
};

struct c{
   struct a beatles;
   struct b beatles2;
};


void main(){

    //Change type from "struct c" to "singMe"
    typedef struct c singMe;

    //Creates Variable
        //struct c AGoodSong;
        //OR
        singMe AGoodSong;

    //Setting values
    AGoodSong.beatles.HelloHello = 1;
    AGoodSong.beatles2.HelloHello = 2;
    AGoodSong.beatles2.YouSayGoodbye = 3;

    //Read Value
    printf("%d\n", AGoodSong.beatles.HelloHello);
    printf("%d\n", AGoodSong.beatles2.HelloHello);
    printf("%d\n", AGoodSong.beatles2.YouSayGoodbye);

}