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

public class EClientSocket_faMsgTypeName_0_0_Test {

    private EClientSocket eClientSocket;

    @BeforeEach
    public void setUp() {
        eClientSocket = new EClientSocket();
    }

    @Test
    public void testFaMsgTypeName() {
        int faDataType = 1;
        String expectedResult = "REQ_OPEN_ORDERS";
        String actualResult = eClientSocket.faMsgTypeName(faDataType);
        assertEquals(expectedResult, actualResult);
    }
}
