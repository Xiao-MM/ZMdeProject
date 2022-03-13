package algorithm;

import java.util.Arrays;

public class SurroundedArea {
    int[][] dirs = new int[][]{{-1,0},{1,0},{0,-1},{0,1}};

    /**
     * 130. 被围绕的区域
     * @param board
     */
    public void solve(char[][] board) {
        int m = board.length;
        int n = board[0].length;
        // 将未被包围的置为T
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if ((i == 0 || i == m - 1 || j == 0 || j == n - 1) && board[i][j] == 'O'){
                    dfs(board, i, j);
                }
            }
        }
        // 将被包围的O置为X，复原T为O
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == 'O') board[i][j] = 'X';
                if (board[i][j] == 'T') board[i][j] = 'O';
            }

        }

    }

    private void dfs(char[][] board, int x, int y){
        board[x][y] = 'T';
        for (int[] dir : dirs) {
            int i = x + dir[0];
            int j = y + dir[1];
            if (i >= 0 && i < board.length && j >= 0 && j < board[0].length && board[i][j] == 'O'){
                dfs(board, i, j);
            }
        }
    }

    public static void main(String[] args) {
        SurroundedArea surroundedArea = new SurroundedArea();
        char[][] board = {
                {'X', 'X', 'X', 'X'},
                {'X', 'O', 'O', 'X'},
                {'X', 'X', 'O', 'X'},
                {'X', 'O', 'X', 'X'}
        };
        surroundedArea.solve(board);
        for (char[] chars : board) {
            System.out.println(Arrays.toString(chars));
        }

    }
}
