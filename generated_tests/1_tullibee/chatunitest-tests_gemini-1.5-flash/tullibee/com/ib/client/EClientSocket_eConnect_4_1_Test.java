package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import static com.ib.client.EClientErrors.ALREADY_CONNECTED;
import static com.ib.client.EClientErrors.NO_VALID_ID;
import static com.ib.client.EClientErrors.UPDATE_TWS;
import static com.ib.client.EReader.readInt;
import static com.ib.client.EReader.readStr;
import java.net.Socket;
import com.ib.client.EClientErrors.CodeMsgPair;

@ExtendWith(MockitoExtension.class)
public class EClientSocket_eConnect_4_1_Test {

    @Mock
    private Socket m_socket;

    @Mock
    private DataInputStream m_dis;

    @Mock
    private DataOutputStream m_dos;

    private EClientSocket eClientSocket;

    @BeforeEach
    void setUp() {
        eClientSocket = new EClientSocket();
        try {
            Field fieldSocket = EClientSocket.class.getDeclaredField("m_socket");
            fieldSocket.setAccessible(true);
            fieldSocket.set(eClientSocket, m_socket);
            Field fieldDis = EClientSocket.class.getDeclaredField("m_dis");
            fieldDis.setAccessible(true);
            fieldDis.set(eClientSocket, m_dis);
            Field fieldDos = EClientSocket.class.getDeclaredField("m_dos");
            fieldDos.setAccessible(true);
            fieldDos.set(eClientSocket, m_dos);
            Field fieldClientId = EClientSocket.class.getDeclaredField("m_clientId");
            fieldClientId.setAccessible(true);
            fieldClientId.setInt(eClientSocket, 123);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
            org.junit.jupiter.api.Assertions.fail("Failed to set up mock objects");
        }
    }

    @Test
    void testEConnectSuccessVersion3() throws IOException {
        when(m_dis.readInt()).thenReturn(7);
        try {
            eClientSocket.eConnect("localhost", 7496, 123);
        } catch (EException e) {
            fail("Unexpected EException: " + e.getMessage());
        }
        verify(m_dos).writeInt(7);
        verify(m_dos).writeInt(123);
        verify(m_dis).readInt();
    }

    @Test
    void testEConnectSuccessVersion20() throws IOException {
        when(m_dis.readInt()).thenReturn(20);
        when(m_dis.readUTF()).thenReturn("TWS Time");
        try {
            eClientSocket.eConnect("localhost", 7496, 123);
        } catch (EException e) {
            fail("Unexpected EException: " + e.getMessage());
        }
        verify(m_dos).writeInt(7);
        verify(m_dos).writeInt(123);
        verify(m_dis, times(2)).readInt();
        verify(m_dis).readUTF();
    }

    @Test
    void testEConnectVersionTooLow() throws IOException {
        when(m_dis.readInt()).thenReturn(6);
        assertThrows(EException.class, () -> eClientSocket.eConnect("localhost", 7496, 123));
        verify(m_dos).writeInt(7);
        verify(m_dis).readInt();
    }

    @Test
    void testEConnectAlreadyConnected() throws IOException {
        when(m_socket.isConnected()).thenReturn(true);
        assertThrows(EException.class, () -> eClientSocket.eConnect("localhost", 7496, 123));
        verify(m_socket, times(1)).isConnected();
        verify(m_dos, never()).writeInt(anyInt());
        verify(m_dis, never()).readInt();
    }
}
