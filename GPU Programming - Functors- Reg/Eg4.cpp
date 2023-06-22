// functor : Function Objects
// Overloading the operator ()
#include<iostream>
using namespace std;
class mul_x{
	private: int x;
	public: 
	mul_x(int val): x(val){ }// Constructor
	int operator()(int y){ return x * y; }
};
int main(int argc, char *argv[]){
 	mul_x mul50(50); // Create an instance of the functor class 
	int y=atoi(argv[1]);
	int res=mul50(y); // res= 50 * Y
	cout<<"Result is: "<<res<<endl;
}	
