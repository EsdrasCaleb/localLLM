package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
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
import java.io.IOException;
import com.ib.client.EClientErrors.CodeMsgPair;

@ExtendWith(MockitoExtension.class)
public class EClientSocket_faMsgTypeName_0_0_Test {

    private EClientSocket eClientSocket;

    @Mock
    private Socket m_socket;

    @Mock
    private DataOutputStream m_dos;

    @Mock
    private DataInputStream m_dis;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        eClientSocket = new EClientSocket();
        // Use reflection to set private fields
        setPrivateField(eClientSocket, "m_socket", m_socket);
        setPrivateField(eClientSocket, "m_dos", m_dos);
        setPrivateField(eClientSocket, "m_dis", m_dis);
    }

    @Test
    public void testFaMsgTypeName_Groups() throws Exception {
        assertEquals("GROUPS", invokeFaMsgTypeName(EClientSocket.GROUPS));
    }

    @Test
    public void testFaMsgTypeName_Profiles() throws Exception {
        assertEquals("PROFILES", invokeFaMsgTypeName(EClientSocket.PROFILES));
    }

    @Test
    public void testFaMsgTypeName_Aliases() throws Exception {
        assertEquals("ALIASES", invokeFaMsgTypeName(EClientSocket.ALIASES));
    }

    @Test
    public void testFaMsgTypeName_Unknown() throws Exception {
        // 999 is an unknown faDataType
        assertNull(invokeFaMsgTypeName(999));
    }

    private String invokeFaMsgTypeName(int faDataType) throws Exception {
        Method method = EClientSocket.class.getDeclaredMethod("faMsgTypeName", int.class);
        method.setAccessible(true);
        return (String) method.invoke(eClientSocket, faDataType);
    }

    private void setPrivateField(Object object, String fieldName, Object value) throws Exception {
        Field field = object.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(object, value);
    }
}
