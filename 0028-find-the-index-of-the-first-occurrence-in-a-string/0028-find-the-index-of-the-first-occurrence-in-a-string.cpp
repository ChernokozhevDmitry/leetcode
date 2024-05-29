class Solution {
public:
    int strStr(string haystack, string needle) {
        int res = haystack.find(needle);
        if (res == std::string::npos) {
            return -1;
        }
        return res;
    }        
};