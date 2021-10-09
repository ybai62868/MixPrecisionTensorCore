#include <iostream>
#include <vector>
#include <cstring>
#include <string>

using namespace std;
vector<int>res;


void dfs(vector<int>& res, int currentSum, int n) {
    if (currentSum == n) {
        int cnt = 0;
        cout << endl;
        cout << "number of groups: " << res.size() << endl;
        for ( int i = 0; i < res.size();i++ ) {
            for ( int j = 0;j < res[i];j++ ) {
                cout << "stream number: " << cnt << " " << endl;;
            }
            cnt++;
        }
        cout << n << " = " << res[0];
        for ( int i = 1; i < res.size();i++ ) {
            cout << " + " << res[i]; 
        }
        cout << endl;
        return;
    }
    for ( int i = n - currentSum;i >= 1;i-- ) {
        if (res.size() > 0 && res.back() < i) {
            continue;
        }
        res.push_back(i);
        currentSum += i;
        dfs(res, currentSum, n);
        res.pop_back();
        currentSum -= i;
    }
}

int main(void)
{
    // int n; cin >> n;
    int in_channels[] = {64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024};
    for ( int i = 0;i < sizeof(in_channels)/sizeof(int);i++ ) {
        dfs(res, 0, in_channels[i]);
    }
    // dfs(res, 0, n);
    return 0;
}
