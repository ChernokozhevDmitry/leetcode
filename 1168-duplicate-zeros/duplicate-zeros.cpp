class Solution {
public:
    void duplicateZeros(vector<int>& arr) {
        for (int i = 0; i < arr.size(); ++i) {
            if (arr[i] == 0) {
                arr.resize(arr.size() - 1);
                arr.insert(arr.begin() + i, 0);
                ++i;
                std::cout << "   ++i = " << i << std::endl;
            }
        }
    }
};