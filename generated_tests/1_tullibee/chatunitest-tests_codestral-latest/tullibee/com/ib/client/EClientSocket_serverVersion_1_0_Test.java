package com.ib.client;

import java.io.IOException;
import java.lang.reflect.Field;
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
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.net.Socket;
import com.ib.client.EClientErrors.CodeMsgPair;

class EClientSocket_serverVersion_1_0_Test {

    @InjectMocks
    private EClientSocket eClientSocket;

    @Mock
    private Socket m_socket;

    @BeforeEach
    void setUp() throws NoSuchFieldException, IllegalAccessException {
        MockitoAnnotations.openMocks(this);
        Field socketField = EClientSocket.class.getDeclaredField("m_socket");
        socketField.setAccessible(true);
        socketField.set(eClientSocket, m_socket);
    }

    @Test
    void testServerVersionConnected() throws EException, NoSuchFieldException, IllegalAccessException {
        when(m_socket.isConnected()).thenReturn(true);
        Field serverVersionField = EClientSocket.class.getDeclaredField("m_serverVersion");
        serverVersionField.setAccessible(true);
        serverVersionField.set(eClientSocket, 38);
        int version = eClientSocket.serverVersion();
        assertEquals(38, version);
    }

    @Test
    void testServerVersionNotConnected() {
        when(m_socket.isConnected()).thenReturn(false);
        assertThrows(EException.class, () -> eClientSocket.serverVersion());
    }
}
