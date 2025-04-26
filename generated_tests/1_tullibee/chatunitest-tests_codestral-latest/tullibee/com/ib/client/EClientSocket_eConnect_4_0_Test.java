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
import static com.ib.client.EReader.readInt;
import static com.ib.client.EReader.readStr;
import com.ib.client.EClientErrors.CodeMsgPair;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class EClientSocket_eConnect_4_0_Test {

    @Mock
    private Socket mockSocket;

    @Mock
    private DataInputStream mockDataInputStream;

    @Mock
    private DataOutputStream mockDataOutputStream;

    @InjectMocks
    private EClientSocket eClientSocket;

    @BeforeEach
    public void setUp() throws IOException {
        when(mockSocket.getInputStream()).thenReturn(mockDataInputStream);
        when(mockSocket.getOutputStream()).thenReturn(mockDataOutputStream);
    }

    @Test
    public void testEConnect_Success() throws IOException, EException {
        when(mockDataInputStream.readInt()).thenReturn(46);
        when(mockDataInputStream.readUTF()).thenReturn("TWS Time");
        eClientSocket.eConnect("localhost", 7496, 1);
        verify(mockSocket).connect(any(), anyInt());
        verify(mockDataOutputStream).writeInt(46);
        verify(mockDataOutputStream).writeInt(1);
    }

    @Test
    public void testEConnect_ServerVersionBelow20() throws IOException, EException {
        when(mockDataInputStream.readInt()).thenReturn(19);
        assertThrows(EException.class, () -> eClientSocket.eConnect("localhost", 7496, 1));
    }

    @Test
    public void testEConnect_ServerVersionBelow3() throws IOException, EException {
        when(mockDataInputStream.readInt()).thenReturn(2);
        eClientSocket.eConnect("localhost", 7496, 1);
        verify(mockSocket).connect(any(), anyInt());
        verify(mockDataOutputStream).writeInt(46);
        verify(mockDataOutputStream, never()).writeInt(1);
    }

    @Test
    public void testEConnect_AlreadyConnected() throws IOException, EException {
        when(eClientSocket.isConnected()).thenReturn(true);
        assertThrows(EException.class, () -> eClientSocket.eConnect("localhost", 7496, 1));
    }

    @Test
    public void testEConnect_ServerVersionTooLow() throws IOException, EException {
        when(mockDataInputStream.readInt()).thenReturn(37);
        assertThrows(EException.class, () -> eClientSocket.eConnect("localhost", 7496, 1));
    }
}
