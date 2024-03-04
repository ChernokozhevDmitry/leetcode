class Solution {
public:
    vector<int> replaceElements(vector<int>& arr) {
        for (int i = 0; i < arr.size() - 1; ++i){
            int max_right = arr[i+1];
            for (int j = i+1; j < arr.size(); ++j){
              max_right = std::max(arr[j], max_right);
            }
            arr[i] = max_right;
        }
        arr[arr.size() - 1] = -1;
        return arr;    
    }
};