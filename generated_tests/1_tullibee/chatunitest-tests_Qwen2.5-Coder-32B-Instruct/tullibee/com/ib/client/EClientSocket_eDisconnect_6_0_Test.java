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

class EClientSocket_eDisconnect_6_0_Test {

    @Mock
    private Socket m_socket;

    @Mock
    private DataOutputStream m_dos;

    @Mock
    private DataInputStream m_dis;

    private EClientSocket eClientSocket;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        eClientSocket = new EClientSocket();
        // Set the private fields using reflection
        setField(eClientSocket, "m_socket", m_socket);
        setField(eClientSocket, "m_dos", m_dos);
        setField(eClientSocket, "m_dis", m_dis);
        setField(eClientSocket, "m_serverVersion", 46);
        setField(eClientSocket, "m_TwsTime", "12:34:56");
    }

    @Test
    void testEDisconnect() throws IOException {
        // Call the method under test
        eClientSocket.eDisconnect();
        // Verify that the socket, dos, dis, serverVersion, and TwsTime are set to null or 0
        assertNull(getField(eClientSocket, "m_socket"));
        assertNull(getField(eClientSocket, "m_dos"));
        assertNull(getField(eClientSocket, "m_dis"));
        assertEquals(0, getField(eClientSocket, "m_serverVersion"));
        assertNull(getField(eClientSocket, "m_TwsTime"));
        // Verify that the socket was closed
        verify(m_socket, times(1)).close();
    }

    @Test
    void testEDisconnectWithNullSocket() throws IOException {
        // Set the socket to null
        setField(eClientSocket, "m_socket", null);
        // Call the method under test
        eClientSocket.eDisconnect();
        // Verify that the socket, dos, dis, serverVersion, and TwsTime are set to null or 0
        assertNull(getField(eClientSocket, "m_socket"));
        assertNull(getField(eClientSocket, "m_dos"));
        assertNull(getField(eClientSocket, "m_dis"));
        assertEquals(0, getField(eClientSocket, "m_serverVersion"));
        assertNull(getField(eClientSocket, "m_TwsTime"));
        // Verify that the socket was not closed
        verify(m_socket, never()).close();
    }

    @Test
    void testEDisconnectWithIOException() throws IOException {
        // Set the socket to throw an IOException when closed
        doThrow(new IOException("Mock IOException")).when(m_socket).close();
        // Call the method under test and expect an IOException
        assertThrows(IOException.class, () -> eClientSocket.eDisconnect());
        // Verify that the socket, dos, dis, serverVersion, and TwsTime are set to null or 0
        assertNull(getField(eClientSocket, "m_socket"));
        assertNull(getField(eClientSocket, "m_dos"));
        assertNull(getField(eClientSocket, "m_dis"));
        assertEquals(0, getField(eClientSocket, "m_serverVersion"));
        assertNull(getField(eClientSocket, "m_TwsTime"));
        // Verify that the socket was closed
        verify(m_socket, times(1)).close();
    }

    private void setField(EClientSocket obj, String fieldName, Object value) throws RuntimeException {
        try {
            var field = EClientSocket.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            field.set(obj, value);
        } catch (Exception e) {
            throw new RuntimeException("Failed to set field: " + fieldName, e);
        }
    }

    private Object getField(EClientSocket obj, String fieldName) throws RuntimeException {
        try {
            var field = EClientSocket.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            return field.get(obj);
        } catch (Exception e) {
            throw new RuntimeException("Failed to get field: " + fieldName, e);
        }
    }
}
