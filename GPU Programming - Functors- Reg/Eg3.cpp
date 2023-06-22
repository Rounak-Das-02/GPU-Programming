// functor : Function Objects
// Overloading the operator ()
#include<iostream>
#include<bits/stdc++.h>
using namespace std;
class add_x{
	private: int x;
	public: 
	add_x(int val): x(val){ }// Constructor
	int operator()(int y) const{ return x + y; }
};
class print_x{
	public:
	void operator()(int x){ cout<<"\t"<<x; }
};
int main(int argc, char *argv[]){
 	std::vector<int> in;
	in.push_back(1); in.push_back(2); in.push_back(3); in.push_back(4);
	std::vector<int> out(in.size());
	std::transform(in.begin(), in.end(), out.begin(), add_x(atoi(argv[1])));
	for_each(out.begin(), out.end(), print_x());
	cout<<endl;
}	
