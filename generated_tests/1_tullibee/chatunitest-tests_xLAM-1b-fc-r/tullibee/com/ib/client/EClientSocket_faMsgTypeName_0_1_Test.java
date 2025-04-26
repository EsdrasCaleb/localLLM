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

public class EClientSocket_faMsgTypeName_0_1_Test {

    @Test
    public void testFaMsgTypeName() {
        // Test with various integer values
        assertEquals("FA_MSG_TYPE_1", EClientSocket.faMsgTypeName(1));
        assertEquals("FA_MSG_TYPE_2", EClientSocket.faMsgTypeName(2));
        assertEquals("FA_MSG_TYPE_3", EClientSocket.faMsgTypeName(3));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_4", EClientSocket.faMsgTypeName(4));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_5", EClientSocket.faMsgTypeName(5));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_6", EClientSocket.faMsgTypeName(6));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_7", EClientSocket.faMsgTypeName(7));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_8", EClientSocket.faMsgTypeName(8));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_9", EClientSocket.faMsgTypeName(9));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_10", EClientSocket.faMsgTypeName(10));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_11", EClientSocket.faMsgTypeName(11));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_12", EClientSocket.faMsgTypeName(12));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_13", EClientSocket.faMsgTypeName(13));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_14", EClientSocket.faMsgTypeName(14));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_15", EClientSocket.faMsgTypeName(15));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_16", EClientSocket.faMsgTypeName(16));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_17", EClientSocket.faMsgTypeName(17));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_18", EClientSocket.faMsgTypeName(18));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_19", EClientSocket.faMsgTypeName(19));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_20", EClientSocket.faMsgTypeName(20));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_21", EClientSocket.faMsgTypeName(21));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_22", EClientSocket.faMsgTypeName(22));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_23", EClientSocket.faMsgTypeName(23));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_24", EClientSocket.faMsgTypeName(24));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_25", EClientSocket.faMsgTypeName(25));
        // Test with a value outside the expected range
        assertEquals("FA_MSG_TYPE_26", EClientSocket.faMsgTypeName(26));
        // Test with a value outside the expected range
    }
}
