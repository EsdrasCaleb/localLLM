package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.IOException;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import static com.ib.client.EClientErrors.ALREADY_CONNECTED;
import static com.ib.client.EClientErrors.NO_VALID_ID;
import static com.ib.client.EClientErrors.UPDATE_TWS;
import static com.ib.client.EReader.readInt;
import static com.ib.client.EReader.readStr;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.net.Socket;
import com.ib.client.EClientErrors.CodeMsgPair;

@ExtendWith(MockitoExtension.class)
public class EClientSocket_faMsgTypeName_0_2_Test {

    @Mock
    private DataOutputStream m_dos;

    @Mock
    private DataInputStream m_dis;

    @InjectMocks
    private EClientSocket m_clientSocket;

    @Test
    public void testFaMsgTypeName() throws IOException {
        // Given
        int faDataType = 10;
        // When
        String result = m_clientSocket.faMsgTypeName(faDataType);
        // Then
        assertEquals("FUNDAMENTAL_DATA", result);
    }
}
