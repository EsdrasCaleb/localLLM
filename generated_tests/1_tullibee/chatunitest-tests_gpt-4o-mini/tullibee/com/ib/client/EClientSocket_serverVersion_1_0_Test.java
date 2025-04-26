package com.ib.client;

import java.io.IOException;
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

    private EClientSocket eClientSocket;

    @BeforeEach
    void setUp() {
        eClientSocket = new EClientSocket();
    }

    @Test
    void testServerVersion_WhenNotConnected_ShouldThrowEException() {
        // Arrange
        // Mocking the isConnected method to return false
        EClientSocket spyClientSocket = spy(eClientSocket);
        doReturn(false).when(spyClientSocket).isConnected();
        // Act & Assert
        EException exception = assertThrows(EException.class, spyClientSocket::serverVersion);
        assertEquals(EClientErrors.NOT_CONNECTED, exception.getErrorCode());
    }

    @Test
    void testServerVersion_WhenConnected_ShouldReturnServerVersion() throws EException {
        // Arrange
        // Mocking the isConnected method to return true
        EClientSocket spyClientSocket = spy(eClientSocket);
        doReturn(true).when(spyClientSocket).isConnected();
        // Setting server version directly using reflection
        try {
            java.lang.reflect.Field field = EClientSocket.class.getDeclaredField("m_serverVersion");
            field.setAccessible(true);
            // Set server version to 42
            field.set(spyClientSocket, 42);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Reflection failed: " + e.getMessage());
        }
        // Act
        int serverVersion = spyClientSocket.serverVersion();
        // Assert
        assertEquals(42, serverVersion);
    }
}
