class Solution {
public:
    bool validMountainArray(vector<int>& arr) {
    bool up = true;
    if (arr.size() < 3) {
        return false;
    }
    else if (arr[0] >= arr [1]) {
        return false;
    }
    for (int i = 2; i < arr.size(); ++i){
        if (arr[i] == arr[i-1]) {
            std::cout << "arr[i] == arr[i-1]" << std::endl;
            return false;
        }
        if (up && (arr[i] < arr[i-1])) {
                up = false;
                continue;
        }
        if (!up && (arr[i] > arr[i-1])) {
                return false;
        }
    }
        return !up;
    }
};