#include<iostream>
using namespace std;

void printX(char const *str){ cout<<"Hello World "<<str<<endl; }

int main(){

	void(*fun)(char const *);
	auto f=printX;
	f("Hi Welcome");
	fun=printX;
	fun("How are you");
	return 0;
}
