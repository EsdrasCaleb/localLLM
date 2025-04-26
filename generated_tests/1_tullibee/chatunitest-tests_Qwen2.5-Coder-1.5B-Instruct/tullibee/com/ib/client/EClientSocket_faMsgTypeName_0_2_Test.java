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

class EClientSocket_faMsgTypeName_0_2_Test {

    @Test
    void testFaMsgTypeName() {
        // Test cases to verify the correctness of the faMsgTypeName method
        assertEquals("GROUPS", EClientSocket.faMsgTypeName(EClientSocket.GROUPS));
        assertEquals("PROFILES", EClientSocket.faMsgTypeName(EClientSocket.PROFILES));
        assertEquals("ALIASES", EClientSocket.faMsgTypeName(EClientSocket.ALIASES));
        // Unknown data type
        assertNull(EClientSocket.faMsgTypeName(10));
    }
}
