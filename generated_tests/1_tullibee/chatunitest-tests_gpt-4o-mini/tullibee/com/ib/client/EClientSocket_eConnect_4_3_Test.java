package com.ib.client;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.Socket;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import static com.ib.client.EClientErrors.ALREADY_CONNECTED;
import static com.ib.client.EClientErrors.NO_VALID_ID;
import static com.ib.client.EClientErrors.UPDATE_TWS;
import com.ib.client.EClientErrors.CodeMsgPair;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import static com.ib.client.EReader.readInt;
import static com.ib.client.EReader.readStr;

class EClientSocket_eConnect_4_3_Test {

    private EClientSocket eClientSocket;

    private Socket mockSocket;

    private DataOutputStream mockDos;

    private DataInputStream mockDis;

    @BeforeEach
    void setUp() throws Exception {
        eClientSocket = new EClientSocket();
        mockSocket = mock(Socket.class);
        mockDos = mock(DataOutputStream.class);
        mockDis = mock(DataInputStream.class);
        // Use reflection to set private fields
        setPrivateField(eClientSocket, "m_socket", mockSocket);
        setPrivateField(eClientSocket, "m_dos", mockDos);
        setPrivateField(eClientSocket, "m_dis", mockDis);
    }

    @Test
    void testEConnect_Success() throws Exception {
        // Mock server version
        when(mockDis.readInt()).thenReturn(40);
        // Mock TWS time
        when(mockDis.readUTF()).thenReturn("TWS Time");
        // Access CLIENT_VERSION through reflection
        int clientVersion = (int) getPrivateField(EClientSocket.class, "CLIENT_VERSION");
        eClientSocket.eConnect("localhost", 7496, 123);
        verify(mockDos).writeInt(clientVersion);
        // Verify clientId is sent
        verify(mockDos).writeInt(123);
        assertEquals("TWS Time", getPrivateField(eClientSocket, "m_TwsTime"));
    }

    @Test
    void testEConnect_AlreadyConnected() throws Exception {
        // Simulate already connected
        when(eClientSocket.isConnected()).thenReturn(true);
        Exception exception = assertThrows(EException.class, () -> {
            eClientSocket.eConnect("localhost", 7496, 123);
        });
        assertEquals(ALREADY_CONNECTED, exception.getMessage());
    }

    @Test
    void testEConnect_ServerVersionTooLow() throws Exception {
        // Mock server version
        when(mockDis.readInt()).thenReturn(30);
        Exception exception = assertThrows(EException.class, () -> {
            eClientSocket.eConnect("localhost", 7496, 123);
        });
        assertEquals(UPDATE_TWS, exception.getMessage());
    }

    @Test
    void testEConnect_ServerVersionNoTime() throws Exception {
        when(mockDis.readInt()).thenReturn(20);
        // Mock TWS time
        when(mockDis.readUTF()).thenReturn("TWS Time");
        int clientVersion = (int) getPrivateField(EClientSocket.class, "CLIENT_VERSION");
        eClientSocket.eConnect("localhost", 7496, 123);
        verify(mockDos).writeInt(clientVersion);
        // clientId should not be sent
        verify(mockDos, never()).writeInt(123);
    }

    private void setPrivateField(Object target, String fieldName, Object value) throws Exception {
        var field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }

    private Object getPrivateField(Object target, String fieldName) throws Exception {
        var field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        return field.get(target);
    }

    private Object getPrivateField(Class<?> clazz, String fieldName) throws Exception {
        var field = clazz.getDeclaredField(fieldName);
        field.setAccessible(true);
        return field.get(null);
    }
}
