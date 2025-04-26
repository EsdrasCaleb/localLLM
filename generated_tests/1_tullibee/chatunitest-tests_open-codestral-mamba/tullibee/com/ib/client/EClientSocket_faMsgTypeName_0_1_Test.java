package com.ib.client;

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

class EClientSocket_faMsgTypeName_0_1_Test {

    @Test
    void testFaMsgTypeName() {
        EClientSocket eClientSocket = new EClientSocket();
        // Test when input is GROUPS
        assertEquals("GROUPS", eClientSocket.faMsgTypeName(EClientSocket.GROUPS));
        // Test when input is PROFILES
        assertEquals("PROFILES", eClientSocket.faMsgTypeName(EClientSocket.PROFILES));
        // Test when input is ALIASES
        assertEquals("ALIASES", eClientSocket.faMsgTypeName(EClientSocket.ALIASES));
        // Test when input is not a valid constant
        assertEquals(null, eClientSocket.faMsgTypeName(100));
    }
}
