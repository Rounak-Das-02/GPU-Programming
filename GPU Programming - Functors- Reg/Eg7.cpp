#include<iostream>
#include<vector>
using namespace std;

void printX(int val){ cout<<"\t"<<val; }
void forEach(vector<int> &values, void(*f)(int))
{
	for(int value:values){ f(value); }
}

int main(){

	vector<int> V1={2,3,4,1,5};
	//forEach(V1, printX);
	forEach(V1, [](int val){ cout<<"\t"<<val; });
	return 0;
}
