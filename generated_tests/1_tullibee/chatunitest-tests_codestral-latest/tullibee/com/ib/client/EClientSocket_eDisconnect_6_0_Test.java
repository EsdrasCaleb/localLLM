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

class EClientSocket_eDisconnect_6_0_Test {

    @Mock
    private Socket m_socket;

    @Mock
    private DataOutputStream m_dos;

    @Mock
    private DataInputStream m_dis;

    @InjectMocks
    private EClientSocket eClientSocket;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testEDisconnect() throws IOException {
        // Set private fields using reflection
        setPrivateField(eClientSocket, "m_socket", m_socket);
        setPrivateField(eClientSocket, "m_dos", m_dos);
        setPrivateField(eClientSocket, "m_dis", m_dis);
        setPrivateField(eClientSocket, "m_serverVersion", 1);
        setPrivateField(eClientSocket, "m_TwsTime", "someTime");
        eClientSocket.eDisconnect();
        verify(m_socket).close();
        assertNull(getPrivateField(eClientSocket, "m_socket"));
        assertNull(getPrivateField(eClientSocket, "m_dos"));
        assertNull(getPrivateField(eClientSocket, "m_dis"));
        assertEquals(0, getPrivateField(eClientSocket, "m_serverVersion"));
        assertNull(getPrivateField(eClientSocket, "m_TwsTime"));
    }

    private void setPrivateField(Object obj, String fieldName, Object value) {
        try {
            var field = obj.getClass().getDeclaredField(fieldName);
            field.setAccessible(true);
            field.set(obj, value);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private Object getPrivateField(Object obj, String fieldName) {
        try {
            var field = obj.getClass().getDeclaredField(fieldName);
            field.setAccessible(true);
            return field.get(obj);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
