package com.ib.client;

import java.io.IOException;
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
import java.net.Socket;
import com.ib.client.EClientErrors.CodeMsgPair;

public class EClientSocket_eDisconnect_6_1_Test {

    @Mock
    private EClientSocket eClientSocket;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @AfterEach
    public void tearDown() {
        eClientSocket = null;
    }

    @Test
    public void eDisconnectTest() throws IOException {
        // Given
        // When
        assertDoesNotThrow(() -> eClientSocket.eDisconnect());
        // Then
        verify(eClientSocket).eDisconnect();
    }
}
