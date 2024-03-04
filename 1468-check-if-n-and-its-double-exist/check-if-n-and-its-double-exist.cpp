class Solution {
public:
    bool checkIfExist(vector<int>& arr) {
        for (int i = 0; i < arr.size(); ++i){
            for (int j = 1; j < arr.size(); ++j){
            if ((i != j) && ((arr[i] == 2 * arr[j]) || ((arr[j] % 2 == 0) && (arr[i] == arr[j] / 2)))) {
                     return true;
                }
            }
        }
        return false;
    }
};