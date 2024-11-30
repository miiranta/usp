#include <iostream>

struct StackNode;
typedef StackNode (*StackPointer);
typedef int entryType;

struct StackNode{
    entryType entry;
    StackPointer nextNode;
};

