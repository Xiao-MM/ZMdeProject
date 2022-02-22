package algorithm;

public class WordSearch {

    int[][] dirs = new int[][]{{-1,0},{1,0},{0,-1},{0,1}};

    String word;
    int[][] isVisited;

    /**
     * 79. 单词搜索
     * @param board
     * @param word
     * @return
     */
    public boolean exist(char[][] board, String word) {
        this.word = word;
        this.isVisited = new int[board.length][board[0].length];
        if (!preCheck(board, word.toCharArray())){
            return false;
        }
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (dfs(board, i, j, 0))
                    return true;
            }
        }
        return false;
    }

    /**
     * 这里首先进行预先检查，快速判断不符合的情况：
     * （1）board中的字符长度小于word的情况
     * （2）board中存在某个字符个数小于word中对应字符情况
     * （3）word中相连两个字符在board中不存在的情况
     */
    private boolean preCheck(char[][] board, char[] word) {
        int m = board.length;
        int n = board[0].length;
        // 情况1
        if(m * n < word.length)
            return false;

        // 情况二
        int[] wc = new int[128];
        for (char[] chars : board) {
            for (int j = 0; j < n; j++) {
                wc[chars[j]]++;
            }
        }
        for(char c : word) {
            wc[c]--;
            if(wc[c] < 0) return false;
        }

        // 情况3
        boolean[][] adj = new boolean[128][128];
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                char a = board[i][j];
                if(i > 0) {
                    char b = board[i-1][j];
                    adj[a][b] = adj[b][a] = true;
                }
                if(j > 0) {
                    char b = board[i][j-1];
                    adj[a][b] = adj[b][a] = true;
                }
            }
        }
        for(int i = 1; i < word.length; i++) {
            if(!adj[word[i-1]][word[i]]) {
                return false;
            }
        }
        return true;
    }

    private boolean dfs(char[][] board, int x, int y, int index){
        if (board[x][y] != word.charAt(index)) return false;
        if (index == word.length() - 1)  return true;
        isVisited[x][y] = 1;
        boolean result = false;
        for (int[] dir : dirs) {
            int i = x + dir[0];
            int j = y + dir[1];
            if (i >= 0 && i < board.length && j >= 0 && j < board[0].length && isVisited[i][j] == 0){
                if (dfs(board, i, j, index + 1)){
                    result = true;
                    break;
                }
            }
        }
        isVisited[x][y] = 0;
        return result;
    }
}
