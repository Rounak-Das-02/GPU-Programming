// functor : Function Objects
// Overloading the operator ()
#include<iostream>
#define string const char *
using namespace std;
class X{
   public: 
   void operator()(string str1, string str2){ 
	   cout<<"Calling functor X with parameter "<<str1<<"\t"<<str2<<endl;
   }
};
int main(int argc, char *argv[]){
 	X foo;
	foo("Hi", "How are you");
	foo("Hello", "I am fine");
	foo(argv[1], argv[2]);
 	return 0;
}	
