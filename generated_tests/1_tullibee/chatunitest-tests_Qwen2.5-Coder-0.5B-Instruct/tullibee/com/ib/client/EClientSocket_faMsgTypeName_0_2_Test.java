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
    void faMsgTypeName() {
        // Arrange
        // This should return null
        int faDataType = 0;
        // Act
        String result = EClientSocket.faMsgTypeName(faDataType);
        // Assert
        assertEquals(null, result);
    }

    @Test
    void faMsgTypeName_1() {
        // Arrange
        // This should return "REQ_MKT_DATA"
        int faDataType = 1;
        // Act
        String result = EClientSocket.faMsgTypeName(faDataType);
        // Assert
        assertEquals("REQ_MKT_DATA", result);
    }

    @Test
    void faMsgTypeName_2() {
        // Arrange
        // This should return "CANCEL_MKT_DATA"
        int faDataType = 2;
        // Act
        String result = EClientSocket.faMsgTypeName(faDataType);
        // Assert
        assertEquals("CANCEL_MKT_DATA", result);
    }
}
