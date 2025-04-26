package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.Socket;
import static com.ib.client.EClientErrors.*;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import static com.ib.client.EReader.readInt;
import static com.ib.client.EReader.readStr;

@ExtendWith(MockitoExtension.class)
public class EClientSocket_serverVersion_1_0_Test {

    @Mock
    private Socket m_socket;

    @InjectMocks
    private EClientSocket eClientSocket;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testServerVersion_NotConnected() throws IOException, EException {
        // Arrange
        when(m_socket.isConnected()).thenReturn(false);
        // Act & Assert
        EException exception = assertThrows(EException.class, () -> {
            eClientSocket.serverVersion();
        });
        assertEquals(NO_VALID_ID, exception.getErrorCode());
        assertEquals(NOT_CONNECTED, exception.getMessage().split("\\[")[0].trim());
    }

    @Test
    public void testServerVersion_Connected() throws IOException, EException, NoSuchFieldException, IllegalAccessException {
        // Arrange
        when(m_socket.isConnected()).thenReturn(true);
        Field serverVersionField = EClientSocket.class.getDeclaredField("m_serverVersion");
        serverVersionField.setAccessible(true);
        serverVersionField.set(eClientSocket, 46);
        // Act
        int serverVersion = eClientSocket.serverVersion();
        // Assert
        assertEquals(46, serverVersion);
    }

    private boolean invokeIsConnected() throws NoSuchMethodException, IllegalAccessException, InvocationTargetException {
        Method isConnectedMethod = EClientSocket.class.getDeclaredMethod("isConnected");
        isConnectedMethod.setAccessible(true);
        return (boolean) isConnectedMethod.invoke(eClientSocket);
    }
}
