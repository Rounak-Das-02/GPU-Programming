#include <algorithm>
#include <vector>
#include<iostream>

using namespace std;

int main() {
	  vector<int> numbers = {1, 2, 3, 4, 5};

	    sort(numbers.begin(), numbers.end(), greater<int>());

	      for (int number : numbers) {
		          cout << number << endl;
			    }

	        return 0;
}

