package com.ib.client;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
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

public class EClientSocket_eDisconnect_6_0_Test {

    private EClientSocket clientSocket;

    private Socket mockSocket;

    private DataOutputStream mockDos;

    private DataInputStream mockDis;

    @BeforeEach
    public void setUp() {
        clientSocket = new EClientSocket();
        mockSocket = mock(Socket.class);
        mockDos = mock(DataOutputStream.class);
        mockDis = mock(DataInputStream.class);
        // Using reflection to set private fields
        setPrivateField(clientSocket, "m_socket", mockSocket);
        setPrivateField(clientSocket, "m_dos", mockDos);
        setPrivateField(clientSocket, "m_dis", mockDis);
        setPrivateField(clientSocket, "m_serverVersion", 1);
        setPrivateField(clientSocket, "m_TwsTime", "2023-10-10T10:00:00Z");
    }

    @Test
    public void testEDisconnect_WithOpenSocket() throws IOException {
        clientSocket.eDisconnect();
        assertNull(getPrivateField(clientSocket, "m_socket"));
        assertNull(getPrivateField(clientSocket, "m_dos"));
        assertNull(getPrivateField(clientSocket, "m_dis"));
        assertEquals(0, getPrivateField(clientSocket, "m_serverVersion"));
        assertNull(getPrivateField(clientSocket, "m_TwsTime"));
        verify(mockSocket).close();
    }

    @Test
    public void testEDisconnect_WithNullSocket() throws IOException {
        // Set the socket to null
        setPrivateField(clientSocket, "m_socket", null);
        clientSocket.eDisconnect();
        assertNull(getPrivateField(clientSocket, "m_socket"));
        assertNull(getPrivateField(clientSocket, "m_dos"));
        assertNull(getPrivateField(clientSocket, "m_dis"));
        assertEquals(0, getPrivateField(clientSocket, "m_serverVersion"));
        assertNull(getPrivateField(clientSocket, "m_TwsTime"));
        verify(mockSocket, never()).close();
    }

    private void setPrivateField(Object obj, String fieldName, Object value) {
        try {
            var field = obj.getClass().getDeclaredField(fieldName);
            field.setAccessible(true);
            field.set(obj, value);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private Object getPrivateField(Object obj, String fieldName) {
        try {
            var field = obj.getClass().getDeclaredField(fieldName);
            field.setAccessible(true);
            return field.get(obj);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
