package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.IOException;
import java.lang.reflect.Field;
import java.net.Socket;
import com.ib.client.*;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import static com.ib.client.EClientErrors.ALREADY_CONNECTED;
import static com.ib.client.EClientErrors.NO_VALID_ID;
import static com.ib.client.EClientErrors.UPDATE_TWS;
import static com.ib.client.EReader.readInt;
import static com.ib.client.EReader.readStr;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import com.ib.client.EClientErrors.CodeMsgPair;

@ExtendWith(MockitoExtension.class)
class EClientSocket_serverVersion_1_0_Test {

    @Mock
    private Socket mockSocket;

    private EClientSocket eClientSocket;

    @BeforeEach
    void setUp() throws NoSuchFieldException, IllegalAccessException {
        eClientSocket = new EClientSocket();
        Field socketField = EClientSocket.class.getDeclaredField("m_socket");
        socketField.setAccessible(true);
        socketField.set(eClientSocket, mockSocket);
    }

    @Test
    void serverVersion_notConnected() throws EException {
        // Mock isConnected to return false
        when(mockSocket.isConnected()).thenReturn(false);
        assertThrows(EException.class, () -> eClientSocket.serverVersion());
    }
}
