class Solution {
public:
    vector<int> plusOne(vector<int>& digits) {
        for (auto i = digits.rbegin(); i != digits.rend(); ++i){
            if (*i < 9){
                (*i)++;
                return digits;
            }
            *i = 0;
        }
        digits[0] = 1;
        digits.push_back(0);
        return digits;
    }
};