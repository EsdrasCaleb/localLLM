package com.ib.client;

import java.lang.reflect.Field;
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

    @Test
    void testFaMsgTypeName() throws Exception {
        EClientSocket ecs = new EClientSocket();
        // Use reflection to set private field m_serverVersion for complete branch coverage.
        Field serverVersionField = EClientSocket.class.getDeclaredField("m_serverVersion");
        serverVersionField.setAccessible(true);
        // Set to a value that will not cause exceptions in other methods
        serverVersionField.setInt(ecs, 40);
        Method faMsgTypeNameMethod = EClientSocket.class.getDeclaredMethod("faMsgTypeName", int.class);
        faMsgTypeNameMethod.setAccessible(true);
        assertEquals("GROUPS", faMsgTypeNameMethod.invoke(ecs, EClientSocket.GROUPS));
        assertEquals("PROFILES", faMsgTypeNameMethod.invoke(ecs, EClientSocket.PROFILES));
        assertEquals("ALIASES", faMsgTypeNameMethod.invoke(ecs, EClientSocket.ALIASES));
        // Test a value outside the defined constants
        assertNull(faMsgTypeNameMethod.invoke(ecs, 0));
        // Test another value outside the defined constants
        assertNull(faMsgTypeNameMethod.invoke(ecs, 100));
    }

    // Helper method to simulate the faMsgTypeName method, allowing for testing without needing the full EClientSocket implementation.
    private String faMsgTypeName(int faDataType) {
        if (faDataType == EClientSocket.GROUPS)
            return "GROUPS";
        if (faDataType == EClientSocket.PROFILES)
            return "PROFILES";
        if (faDataType == EClientSocket.ALIASES)
            return "ALIASES";
        return null;
    }
}
