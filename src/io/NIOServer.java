package io;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.channels.SelectionKey;
import java.nio.channels.Selector;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.util.Iterator;
import java.util.Set;

public class NIOServer {
    public static void main(String[] args) throws IOException {
        // 1. 创建选择器
        Selector selector = Selector.open();

        // 2. 将通道注册到选择器上
        ServerSocketChannel serverSocketChannel = ServerSocketChannel.open();
        serverSocketChannel.configureBlocking(false);// 设置非阻塞
        serverSocketChannel.register(selector, SelectionKey.OP_ACCEPT);

        ServerSocket socket = serverSocketChannel.socket();
        socket.bind(new InetSocketAddress("127.0.0.1", 8888));

        while (true) {
            // 3. 监听事件
            int num = selector.select();
            Set<SelectionKey> selectionKeys = selector.selectedKeys();
            Iterator<SelectionKey> iterator = selectionKeys.iterator();
            while (iterator.hasNext()) {
                SelectionKey key = iterator.next();
                if (key.isAcceptable()) {
                    ServerSocketChannel channel = (ServerSocketChannel) key.channel();
                    // 服务器会为每个新连接创建一个 SocketChannel
                    SocketChannel socketChannel = channel.accept();
                    socketChannel.configureBlocking(false);
                    // 这个新连接主要用于从客户端读取数据
                    socketChannel.register(selector, SelectionKey.OP_READ);
                }
                if (key.isReadable()){
                    SocketChannel channel = (SocketChannel) key.channel();
                    System.out.println(readDataFromSocketChannel(channel));
                    channel.close();
                }
                iterator.remove();
            }
        }
    }

    private static String readDataFromSocketChannel(SocketChannel socketChannel) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(1024);

        StringBuilder data = new StringBuilder();

        while (true) {
            buffer.clear();
            int readNum = socketChannel.read(buffer);
            if (readNum == -1) {
                break;
            }
            // 切换读写
            buffer.flip();
            // 剩余可读
            int limit = buffer.limit();
            char[] chars = new char[limit];
            for (int i = 0; i < limit; i++) {
                chars[i] = (char) buffer.get(i);
            }
            data.append(chars);
            buffer.clear();
        }
        return data.toString();
    }
}