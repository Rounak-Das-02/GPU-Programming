#include<iostream>
#include<bits/stdc++.h>
using namespace std;
int evaluate(int x, int y) { return x*x+2*y+3; }
int main(int argc, char *argv[]){
	set<int> myset={6,7,8,9,10};
	int y=atoi(argv[1]);
	list<int> ps;
	auto f=function<int (int, int)>(evaluate);
transform(myset.begin(), myset.end(), back_inserter(ps), bind(f, placeholders::_1, y));
	for(list<int>::iterator it=ps.begin(); it!=ps.end(); it++)
	{
		cout<<"\t"<<*it;
	}
	cout<<endl;
return 0;
}
