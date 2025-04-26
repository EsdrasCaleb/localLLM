// Test method
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

public class EClientSocket_faMsgTypeName_0_2_Test {

    @Test
    public void faMsgTypeName_1() {
        // Repaired line
        assertEquals("GROUPS", EClientSocket.faMsgTypeName(1));
    }

    @Test
    public void faMsgTypeName_2() {
        // Repaired line
        assertEquals("PROFILES", EClientSocket.faMsgTypeName(2));
    }

    @Test
    public void faMsgTypeName_3() {
        // Repaired line
        assertEquals("ALIASES", EClientSocket.faMsgTypeName(3));
    }
}
