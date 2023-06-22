#include <algorithm>
#include <vector>
#include<iostream>

using namespace std;
class myComp {
	public: bool operator()(int a, int b) const { return a > b; }
};
int main() {
	  vector<int> numbers = {111, 12, 13, 444, 55};

	    sort(numbers.begin(), numbers.end(), myComp());

	      for (int number : numbers) {
		          cout << number << endl;
			    }

	        return 0;
}

