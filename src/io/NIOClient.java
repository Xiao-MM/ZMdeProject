package io;

import java.io.IOException;
import java.io.OutputStream;
import java.net.Socket;
import java.util.concurrent.TimeUnit;

public class NIOClient {
    public static void main(String[] args) throws IOException, InterruptedException {
        for (int i = 0; i < 100; i++) {
            Socket socket = new Socket("127.0.0.1", 8888);
            OutputStream outputStream = socket.getOutputStream();
            TimeUnit.SECONDS.sleep(1);
            String s = "hello world..." + i;
            outputStream.write(s.getBytes());
            outputStream.close();
        }
    }
}
