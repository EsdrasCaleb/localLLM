package com.ib.client;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import java.net.Socket;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import static com.ib.client.EClientErrors.ALREADY_CONNECTED;
import static com.ib.client.EClientErrors.NO_VALID_ID;
import static com.ib.client.EClientErrors.UPDATE_TWS;
import static com.ib.client.EReader.readInt;
import static com.ib.client.EReader.readStr;
import com.ib.client.EClientErrors.CodeMsgPair;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class EClientSocket_eDisconnect_6_1_Test {

    @Test
    void testEDisconnect_SocketNotNull() throws IOException, NoSuchFieldException, IllegalAccessException {
        EClientSocket clientSocket = new EClientSocket();
        Socket mockSocket = mock(Socket.class);
        Field socketField = EClientSocket.class.getDeclaredField("m_socket");
        socketField.setAccessible(true);
        socketField.set(clientSocket, mockSocket);
        Field disField = EClientSocket.class.getDeclaredField("m_dis");
        disField.setAccessible(true);
        Field dosField = EClientSocket.class.getDeclaredField("m_dos");
        dosField.setAccessible(true);
        Field twsTimeField = EClientSocket.class.getDeclaredField("m_TwsTime");
        twsTimeField.setAccessible(true);
        Field serverVersionField = EClientSocket.class.getDeclaredField("m_serverVersion");
        serverVersionField.setAccessible(true);
        disField.set(clientSocket, new DataInputStream(mockSocket.getInputStream()));
        dosField.set(clientSocket, new DataOutputStream(mockSocket.getOutputStream()));
        // Set a test value
        twsTimeField.set(clientSocket, "testTime");
        // set a test value
        serverVersionField.set(clientSocket, 123);
        clientSocket.eDisconnect();
        verify(mockSocket).close();
        assertNull(socketField.get(clientSocket));
        assertNull(disField.get(clientSocket));
        assertNull(dosField.get(clientSocket));
        assertEquals(0, serverVersionField.getInt(clientSocket));
        assertNull(twsTimeField.get(clientSocket));
    }

    @Test
    void testEDisconnect_SocketNull() throws IOException, NoSuchFieldException, IllegalAccessException {
        EClientSocket clientSocket = new EClientSocket();
        Field socketField = EClientSocket.class.getDeclaredField("m_socket");
        socketField.setAccessible(true);
        socketField.set(clientSocket, null);
        Field serverVersionField = EClientSocket.class.getDeclaredField("m_serverVersion");
        serverVersionField.setAccessible(true);
        Field disField = EClientSocket.class.getDeclaredField("m_dis");
        disField.setAccessible(true);
        Field dosField = EClientSocket.class.getDeclaredField("m_dos");
        dosField.setAccessible(true);
        Field twsTimeField = EClientSocket.class.getDeclaredField("m_TwsTime");
        twsTimeField.setAccessible(true);
        clientSocket.eDisconnect();
        assertNull(socketField.get(clientSocket));
        assertNull(disField.get(clientSocket));
        assertNull(dosField.get(clientSocket));
        assertEquals(0, serverVersionField.getInt(clientSocket));
        assertNull(twsTimeField.get(clientSocket));
    }
}
