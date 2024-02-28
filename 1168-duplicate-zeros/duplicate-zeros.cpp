class Solution {
public:
    void duplicateZeros(vector<int>& arr) {
        std::vector<int> newarr(arr.size(),0);
        for (int i = 0, j = 0; i < newarr.size(); ++i, ++j) {
            newarr[i] = arr[j];        
            if (arr[j] == 0) {
                ++i;
            }
        }
        arr = newarr;
    }
};