package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import static com.ib.client.EClientErrors.ALREADY_CONNECTED;
import static com.ib.client.EClientErrors.NO_VALID_ID;
import static com.ib.client.EClientErrors.UPDATE_TWS;
import static com.ib.client.EReader.readInt;
import static com.ib.client.EReader.readStr;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.Socket;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import com.ib.client.EClientErrors.CodeMsgPair;

@ExtendWith(MockitoExtension.class)
public class EClientSocket_TwsConnectionTime_2_1_Test {

    @Mock
    private Socket m_socket;

    @Mock
    private DataOutputStream m_dos;

    @Mock
    private DataInputStream m_dis;

    @InjectMocks
    private EClientSocket eClientSocket;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        setField(eClientSocket, "m_socket", m_socket);
        setField(eClientSocket, "m_dos", m_dos);
        setField(eClientSocket, "m_dis", m_dis);
    }

    private void setField(Object target, String fieldName, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }

    @Test
    public void testTwsConnectionTime_NotConnected() throws Exception {
        setField(eClientSocket, "m_serverVersion", getServerVersion());
        setField(eClientSocket, "m_TwsTime", "12:34:56");
        when(m_socket.isConnected()).thenReturn(false);
        // Act & Assert
        EException exception = assertThrows(EException.class, () -> {
            eClientSocket.TwsConnectionTime();
        });
        assertEquals(NO_VALID_ID, exception.getErrorCode());
        assertEquals(EClientErrors.NOT_CONNECTED.msg(), exception.getMessage());
    }

    @Test
    public void testTwsConnectionTime_Connected() throws Exception {
        setField(eClientSocket, "m_serverVersion", getServerVersion());
        setField(eClientSocket, "m_TwsTime", "12:34:56");
        when(m_socket.isConnected()).thenReturn(true);
        // Act
        String result = eClientSocket.TwsConnectionTime();
        // Assert
        assertEquals("12:34:56", result);
    }

    private int getServerVersion() throws Exception {
        Field serverVersionField = EClientSocket.class.getDeclaredField("SERVER_VERSION");
        serverVersionField.setAccessible(true);
        return (int) serverVersionField.get(null);
    }
}
