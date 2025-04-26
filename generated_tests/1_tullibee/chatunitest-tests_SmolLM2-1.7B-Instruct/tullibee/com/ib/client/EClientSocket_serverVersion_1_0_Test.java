package com.ib.client;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import static org.junit.Assert.*;
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
import java.io.IOException;
import java.net.Socket;
import com.ib.client.EClientErrors.CodeMsgPair;

@RunWith(MockitoJUnitRunner.class)
public class EClientSocket_serverVersion_1_0_Test {

    @Mock
    private EClientSocket clientSocket;

    @InjectMocks
    private EClientSocket underTest;

    @Test
    public void testServerVersion() throws EException {
        // Arrange
        when(clientSocket.isConnected()).thenReturn(true);
        when(clientSocket.serverVersion()).thenReturn(1);
        // Act
        int actualVersion = clientSocket.serverVersion();
        // Assert
        assertEquals(1, actualVersion);
    }
}
