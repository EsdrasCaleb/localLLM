package com.ib.client;

import java.lang.reflect.Method;
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

public class EClientSocket_faMsgTypeName_0_0_Test {

    @InjectMocks
    private EClientSocket eClientSocket;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testFaMsgTypeName() throws Exception {
        Method method = EClientSocket.class.getDeclaredMethod("faMsgTypeName", int.class);
        method.setAccessible(true);
        assertEquals("GROUPS", method.invoke(eClientSocket, 1));
        assertEquals("PROFILES", method.invoke(eClientSocket, 2));
        assertEquals("ALIASES", method.invoke(eClientSocket, 3));
        assertNull(method.invoke(eClientSocket, 0));
        assertNull(method.invoke(eClientSocket, 4));
    }
}
