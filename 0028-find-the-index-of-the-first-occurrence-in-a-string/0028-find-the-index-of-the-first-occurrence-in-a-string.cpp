class Solution {
public:
    int strStr(string haystack, string needle) {
        size_t res = haystack.find(needle);
        return res == std::string::npos ? -1 : res;
    }        
};
