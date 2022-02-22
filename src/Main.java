

import algorithm.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {
    public static void main(String[] args) {
//        System.out.println(StaticMethod.kthSmallest(new int[][]{
//                {1,5,9},
//                {10,11,13},
//                {12,13,15}
//        },8));
//        System.out.println(StaticMethod.uniquePathsWithObstacles(new int[][]{
//                {0, 0, 0},
//                {0, 1, 0},
//                {0, 0, 0}
//                {0,1},{0,0}
//                {1,0}
//        }));
//        System.out.println(StaticMethod.numDecodings("11106"));
//        char a = '0';
//        System.out.println(Integer.valueOf(a));
//        System.out.println(StaticMethod.minPathSum(new int[][]{
////                {1, 3, 1},
////                {1, 5, 1},
////                {4, 2, 1}
//                {1,2,3},{4,5,6}
//        }));
//        System.out.println(StaticMethod.rob(new int[]{2,3,2,3,4}));
//        String a = "";
//        System.out.println(a);
//        System.out.println("".hashCode());
//        System.out.println();
//        System.out.println("" == null);

        WordSearch wordSearch = new WordSearch();
        System.out.println(wordSearch.exist(new char[][]{
                {'A','B','C','E'},
                {'S','F','C','S'},
                {'A','D','E','E'}
        }, "ABCCED"));
    }
}
