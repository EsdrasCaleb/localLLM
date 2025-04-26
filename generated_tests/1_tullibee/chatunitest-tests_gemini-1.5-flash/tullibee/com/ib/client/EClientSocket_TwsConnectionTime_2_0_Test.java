package com.ib.client;

import java.io.*;
import java.lang.reflect.Field;
import java.net.Socket;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import static com.ib.client.EClientErrors.ALREADY_CONNECTED;
import static com.ib.client.EClientErrors.NO_VALID_ID;
import static com.ib.client.EClientErrors.UPDATE_TWS;
import static com.ib.client.EReader.readInt;
import static com.ib.client.EReader.readStr;
import com.ib.client.EClientErrors.CodeMsgPair;

public class EClientSocket_TwsConnectionTime_2_0_Test {

    private EClientSocket eClientSocket;

    private Socket mockSocket;

    private DataInputStream mockDis;

    private DataOutputStream mockDos;

    @BeforeEach
    void setUp() throws Exception {
        mockSocket = Mockito.mock(Socket.class);
        mockDis = Mockito.mock(DataInputStream.class);
        mockDos = Mockito.mock(DataOutputStream.class);
        eClientSocket = new EClientSocket();
        Field socketField = EClientSocket.class.getDeclaredField("m_socket");
        socketField.setAccessible(true);
        socketField.set(eClientSocket, mockSocket);
        Field disField = EClientSocket.class.getDeclaredField("m_dis");
        disField.setAccessible(true);
        disField.set(eClientSocket, mockDis);
        Field dosField = EClientSocket.class.getDeclaredField("m_dos");
        dosField.setAccessible(true);
        dosField.set(eClientSocket, mockDos);
        Field twsTimeField = EClientSocket.class.getDeclaredField("m_TwsTime");
        twsTimeField.setAccessible(true);
        twsTimeField.set(eClientSocket, "2024-10-27 10:00:00");
        Field serverVersionField = EClientSocket.class.getDeclaredField("m_serverVersion");
        serverVersionField.setAccessible(true);
        serverVersionField.set(eClientSocket, 38);
    }

    @Test
    void testTwsConnectionTimeConnected() throws Exception {
        when(mockSocket.isConnected()).thenReturn(true);
        String connectionTime = eClientSocket.TwsConnectionTime();
        assertEquals("2024-10-27 10:00:00", connectionTime);
    }

    @Test
    void testTwsConnectionTimeNotConnected() throws Exception {
        when(mockSocket.isConnected()).thenReturn(false);
        assertThrows(EException.class, () -> eClientSocket.TwsConnectionTime());
    }

    // Dummy EException class for compilation
    static class EException extends Exception {

        public EException() {
            super();
        }

        public EException(String message) {
            super(message);
        }
    }

    // Dummy EClientErrors class for compilation
    static class EClientErrors {

        public static final int NO_VALID_ID = 1;

        public static final int NOT_CONNECTED = 2;
    }
}
