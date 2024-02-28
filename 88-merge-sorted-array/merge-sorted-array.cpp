class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int mm = m;
        for(int j = 0; j!=nums2.size(); ++j){
            nums1[mm] = nums2[j];
            for(int i = mm; i > 0; --i){
                if (nums1[i] < nums1[i-1]) {
                    int temp = nums1[i];
                    nums1[i] = nums1[i-1];
                    nums1[i-1] = temp; 
                }
            }
            ++mm;
        }
    }
};