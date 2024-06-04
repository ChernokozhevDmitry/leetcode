class Solution {
public:
    void reverseString(vector<char>& s) {
        for(std::vector<char>::iterator i = s.begin(), j = s.end() - 1; i <= j; ++i, --j){
            std::swap(*i, *j);
        }
        
    }
};