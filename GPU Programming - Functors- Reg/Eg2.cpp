// functor : Function Objects
// Overloading the operator ()
#include<iostream>
#include<assert.h>
#define string const char *
using namespace std;
class add_x{
	private: int x;
	public: 
	add_x(int val): x(val){ }// Constructor
	int operator()(int y) const{ return x + y; }
};
int main(int argc, char *argv[]){
 	add_x add50(50); // Create an instance of the functor class 
	int y=atoi(argv[1]);
	int res=add50(y); // res=50 +y
	cout<<"\n Result is: "<<res <<endl;
}	
