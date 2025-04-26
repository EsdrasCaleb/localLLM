package com.ib.client;

import java.io.IOException;
import java.net.Socket;
import org.mockito.junit.jupiter.MockitoExtension;
import org.junit.jupiter.api.extension.ExtendWith;
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
class EClientSocket_TwsConnectionTime_2_3_Test {

    private EClientSocket eClientSocket;

    private Socket mockSocket;

    private java.io.OutputStream mockOutputStream;

    private java.io.InputStream mockInputStream;

    @BeforeEach
    void setUp() throws IOException {
        mockSocket = Mockito.mock(Socket.class);
        mockOutputStream = Mockito.mock(java.io.OutputStream.class);
        mockInputStream = Mockito.mock(java.io.InputStream.class);
        when(mockSocket.getOutputStream()).thenReturn(mockOutputStream);
        when(mockSocket.getInputStream()).thenReturn(mockInputStream);
        eClientSocket = new EClientSocket();
        try {
            java.lang.reflect.Field m_socketField = EClientSocket.class.getDeclaredField("m_socket");
            m_socketField.setAccessible(true);
            m_socketField.set(eClientSocket, mockSocket);
            java.lang.reflect.Field m_disField = EClientSocket.class.getDeclaredField("m_dis");
            m_disField.setAccessible(true);
            m_disField.set(eClientSocket, Mockito.mock(DataInputStream.class));
            java.lang.reflect.Field m_dosField = EClientSocket.class.getDeclaredField("m_dos");
            m_dosField.setAccessible(true);
            m_dosField.set(eClientSocket, Mockito.mock(DataOutputStream.class));
            java.lang.reflect.Field m_TwsTimeField = EClientSocket.class.getDeclaredField("m_TwsTime");
            m_TwsTimeField.setAccessible(true);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private fields", e);
        }
    }

    @Test
    void testTwsConnectionTimeConnected() throws EException {
        try {
            java.lang.reflect.Field m_TwsTimeField = EClientSocket.class.getDeclaredField("m_TwsTime");
            m_TwsTimeField.setAccessible(true);
            m_TwsTimeField.set(eClientSocket, "2023-10-27 10:00:00");
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private fields", e);
        }
        when(eClientSocket.isConnected()).thenReturn(true);
        String connectionTime = eClientSocket.TwsConnectionTime();
        assertEquals("2023-10-27 10:00:00", connectionTime);
    }

    @Test
    void testTwsConnectionTimeDisconnected() throws EException {
        when(eClientSocket.isConnected()).thenReturn(false);
        assertThrows(EException.class, () -> eClientSocket.TwsConnectionTime());
    }

    @Test
    void testTwsConnectionTimeNullTime() throws EException {
        try {
            java.lang.reflect.Field m_TwsTimeField = EClientSocket.class.getDeclaredField("m_TwsTime");
            m_TwsTimeField.setAccessible(true);
            m_TwsTimeField.set(eClientSocket, null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private fields", e);
        }
        when(eClientSocket.isConnected()).thenReturn(true);
        String connectionTime = eClientSocket.TwsConnectionTime();
        assertNull(connectionTime);
    }
}
