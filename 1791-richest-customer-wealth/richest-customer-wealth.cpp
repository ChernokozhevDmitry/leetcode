class Solution {
public:
    int maximumWealth(vector<vector<int>>& accounts) {
        int result = 0;
        int sum = 0;
        for (const vector<int>& client : accounts) {
            for (int bank : client) {
                sum += bank; 
            }
            if (sum >= result) {
                result = sum;
            } 
            sum = 0;

        }
        return result;
    }
};