package com.ib.client;

import java.lang.reflect.Field;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.net.Socket;
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
import java.io.IOException;
import com.ib.client.EClientErrors.CodeMsgPair;

@ExtendWith(MockitoExtension.class)
class EClientSocket_TwsConnectionTime_2_0_Test {

    @InjectMocks
    private EClientSocket eClientSocket;

    @Mock
    private Socket m_socket;

    @BeforeEach
    void setUp() throws NoSuchFieldException, IllegalAccessException {
        MockitoAnnotations.openMocks(this);
        Field serverVersionField = EClientSocket.class.getDeclaredField("m_serverVersion");
        serverVersionField.setAccessible(true);
        serverVersionField.set(eClientSocket, 46);
    }

    @Test
    void testTwsConnectionTime_Connected() throws EException, NoSuchFieldException, IllegalAccessException {
        when(m_socket.isConnected()).thenReturn(true);
        Field twsTimeField = EClientSocket.class.getDeclaredField("m_TwsTime");
        twsTimeField.setAccessible(true);
        twsTimeField.set(eClientSocket, "20230925:10:00:00");
        String result = eClientSocket.TwsConnectionTime();
        assertEquals("20230925:10:00:00", result);
    }

    @Test
    void testTwsConnectionTime_NotConnected() {
        when(m_socket.isConnected()).thenReturn(false);
        EException exception = assertThrows(EException.class, () -> {
            eClientSocket.TwsConnectionTime();
        });
        assertEquals(EClientErrors.NO_VALID_ID, exception.getErrorCode());
        assertEquals("Not connected", exception.getMessage());
    }
}
